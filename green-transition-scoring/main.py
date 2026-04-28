import io
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import pandas as pd
import pytesseract
import requests
from PIL import Image, ImageFilter, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent

DATASET_URL = os.getenv("DATASET_URL")
if not DATASET_URL:
    raise ValueError("Please set DATASET_URL as an environment variable.")

DATASET_LOCAL_FILE = None
CACHE_DIR = BASE_DIR / "cache_text"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_CSV_PATH = BASE_DIR / "green_transition_scores_budget_vi.csv"
OUTPUT_XLSX_PATH = BASE_DIR / "green_transition_scores_budget_vi.xlsx"
DETAIL_CSV_PATH = BASE_DIR / "green_transition_indicator_details_budget_vi.csv"
DETAIL_XLSX_PATH = BASE_DIR / "green_transition_indicator_details_budget_vi.xlsx"
ERROR_CSV_PATH = BASE_DIR / "green_transition_errors_budget_vi.csv"

PARTIAL_OUTPUT_PATH = BASE_DIR / "green_transition_scores_budget_vi_partial.csv"
PARTIAL_DETAIL_PATH = BASE_DIR / "green_transition_indicator_details_budget_vi_partial.csv"
PARTIAL_ERROR_PATH = BASE_DIR / "green_transition_errors_budget_vi_partial.csv"

START_ROW = 1
END_ROW = 100000
BATCH_SAVE_EVERY = 50


OCR_ALL_PAGES = False      
OCR_DPI = 240 
OCR_CONFIG = "--oem 1 --psm 3"  
MIN_NATIVE_WORDS_PER_PAGE = 40

CHUNK_WORDS = 150
CHUNK_OVERLAP = 30
TOP_K_PER_INDICATOR = 2
MAX_CHUNK_CHARS_IN_PROMPT = 700
MAX_TOTAL_CHUNKS_PER_BATCH = 10


RUN_JUDGE = True
MAX_JUDGED_INDICATORS_PER_REPORT = 1
MIN_SECONDS_BETWEEN_LLM_CALLS = 0.2
MAX_RETRIES = 5
LLM_MAX_TOKENS_SCORE_BATCH = 1700
LLM_MAX_TOKENS_JUDGE = 500

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-nano").strip()
LLM_MODEL_SCORER = os.getenv("LLM_MODEL_SCORER", LLM_MODEL).strip()
LLM_MODEL_JUDGE = os.getenv("LLM_MODEL_JUDGE", LLM_MODEL_SCORER).strip()


session = requests.Session()
LAST_LLM_CALL_TS = 0.0


# RUBRIC 
RUBRIC_GROUPS = [
    {
        "primary_indicator": "Environmental Management",
        "scoring_instruction": "Disclosure: 2; Not disclosed: 0",
        "scoring_type": "disclosure_binary",
        "secondary_indicators": [
            "Environmental protection concept",
            "Environmental management regime system",
            "Environmental protection special action",
            "Environmental protection time emergency mechanism",
        ],
    },
    {
        "primary_indicator": "Environmental Information Disclosure Carriers",
        "scoring_instruction": "Disclosure: 2; Not disclosed: 0",
        "scoring_type": "disclosure_binary",
        "secondary_indicators": [
            "Annual reports of listed companies",
            "Social resonsibility report",
            "Environmental report",
        ],
    },
    {
        "primary_indicator": "Enviromental Certification",
        "scoring_instruction": "Yes: 2; No: 0",
        "scoring_type": "yes_no_binary",
        "secondary_indicators": [
            "Key pollution monitoring units",
            "Pollutant emissions meet the standard",
            "Whether through the ISO24002 certification",
            "Whether through the ISO9002 certification",
        ],
    },
    {
        "primary_indicator": "Environmental Responsibility",
        "scoring_instruction": "Yes: 2; No: 0",
        "scoring_type": "yes_no_binary",
        "secondary_indicators": [
            "Whether to disclose environmental and sustainable development",
        ],
    },
    {
        "primary_indicator": "Environmental Liabilities",
        "scoring_instruction": "No Description: 0; Qualitative: 1; Quantitative: 2",
        "scoring_type": "description_scale",
        "secondary_indicators": [
            "Wastewater emissions",
            "COD emissions",
            "SO2 emissions",
            "Soot and dust emissions",
            "Industrial solid waste emissions",
        ],
    },
    {
        "primary_indicator": "Environmental Performance and Governance",
        "scoring_instruction": "No Description: 0; Qualitative: 1; Quantitative: 2",
        "scoring_type": "description_scale",
        "secondary_indicators": [
            "Exhaust gas emission reduction treatment",
            "Wastewater abatement treatment",
            "Soot and dust management",
            "Solid waste utilization and disposal",
        ],
    },
    {
        "primary_indicator": "Green Innovation",
        "scoring_instruction": "No Description: 0; Qualitative: 1; Quantitative: 2",
        "scoring_type": "description_scale",
        "secondary_indicators": [
            "whether it has a green invention patent",
            "whether it has a green utility model patent",
        ],
    },
    {
        "primary_indicator": "Environmental Activities",
        "scoring_instruction": "Yes: -1; No: 0",
        "scoring_type": "negative_yes_penalty",
        "secondary_indicators": [
            "environmental violations/accidents/ petitions",
        ],
    },
]

SCORING_RULES = {
    "disclosure_binary": {"allowed_scores": [0, 2]},
    "yes_no_binary": {"allowed_scores": [0, 2]},
    "description_scale": {"allowed_scores": [0, 1, 2]},
    "negative_yes_penalty": {"allowed_scores": [-1, 0]},
}

INDICATOR_HINTS = {
    "Environmental protection concept": ["environmental philosophy", "green concept", "bảo vệ môi trường", "cam kết bảo vệ môi trường", "phát triển bền vững"],
    "Environmental management regime system": ["management system", "policy", "regime", "quy chế", "hệ thống quản lý môi trường", "EMS", "environmental management system"],
    "Environmental protection special action": ["special action", "campaign", "initiative", "chương trình môi trường", "xử lý môi trường", "quản trị môi trường"],
    "Environmental protection time emergency mechanism": ["emergency mechanism", "incident response", "ứng phó khẩn cấp"],
    "Annual reports of listed companies": ["annual report", "báo cáo thường niên"],
    "Social resonsibility report": ["social responsibility report", "csr report", "báo cáo trách nhiệm xã hội"],
    "Environmental report": ["environmental report", "sustainability report", "báo cáo môi trường"],
    "Key pollution monitoring units": ["key pollution monitoring", "monitoring unit", "đơn vị giám sát ô nhiễm trọng điểm"],
    "Pollutant emissions meet the standard": ["meet standard", "compliant", "đạt quy chuẩn", "đạt tiêu chuẩn xả thải"],
    "Whether through the ISO24002 certification": ["ISO24002", "ISO 24002", "certification"],
    "Whether through the ISO9002 certification": ["ISO9002", "ISO 9002", "certification"],
    "Whether to disclose environmental and sustainable development": ["environmental disclosure", "sustainable development disclosure", "công bố môi trường"],
    "Wastewater emissions": ["wastewater", "nước thải", "discharge", "lượng xả thải", "nước sinh hoạt", "nước công nghiệp"],
    "COD emissions": ["COD", "chemical oxygen demand"],
    "SO2 emissions": ["SO2", "sulfur dioxide"],
    "Soot and dust emissions": ["dust", "soot", "bụi", "khói bụi", "bụi và khói"],
    "Industrial solid waste emissions": ["industrial solid waste", "chất thải rắn công nghiệp"],
    "Exhaust gas emission reduction treatment": ["exhaust gas treatment", "air pollution control", "xử lý khí thải", "giảm thiểu khí thải"],
    "Wastewater abatement treatment": ["wastewater treatment", "abatement", "xử lý nước thải", "giảm thiểu nước thải"],
    "Soot and dust management": ["dust management", "soot management", "quản lý bụi", "quản lý khói và bụi"],
    "Solid waste utilization and disposal": ["solid waste utilization", "recycling", "xử lý chất thải rắn"],
    "whether it has a green invention patent": ["green patent", "invention patent", "bằng sáng chế xanh"],
    "whether it has a green utility model patent": ["green utility model patent", "utility model"],
    "environmental violations/accidents/ petitions": ["environmental violation", "accident", "petition", "xử phạt môi trường"],
}


PRIMARY_VI = {
    "Environmental Management": "Quản lý môi trường",
    "Environmental Information Disclosure Carriers": "Kênh công bố thông tin môi trường",
    "Enviromental Certification": "Chứng nhận môi trường",
    "Environmental Responsibility": "Trách nhiệm môi trường",
    "Environmental Liabilities": "Nghĩa vụ môi trường",
    "Environmental Performance and Governance": "Hiệu quả và quản trị môi trường",
    "Green Innovation": "Đổi mới xanh",
    "Environmental Activities": "Sự cố / vi phạm môi trường",
}

SECONDARY_VI = {
    "Environmental protection concept": "Quan điểm / cam kết bảo vệ môi trường",
    "Environmental management regime system": "Hệ thống quản lý môi trường",
    "Environmental protection special action": "Xử lý & quản trị môi trường",
    "Environmental protection time emergency mechanism": "Cơ chế ứng phó khẩn cấp về môi trường",
    "Annual reports of listed companies": "Báo cáo thường niên của công ty niêm yết",
    "Social resonsibility report": "Báo cáo trách nhiệm xã hội",
    "Environmental report": "Báo cáo môi trường",
    "Key pollution monitoring units": "Đơn vị / đối tượng giám sát ô nhiễm trọng điểm",
    "Pollutant emissions meet the standard": "Phát thải chất ô nhiễm đạt chuẩn",
    "Whether through the ISO24002 certification": "Có đạt chứng nhận ISO24002 hay không",
    "Whether through the ISO9002 certification": "Có đạt chứng nhận ISO9002 hay không",
    "Whether to disclose environmental and sustainable development": "Có công bố thông tin về môi trường và phát triển bền vững hay không",
    "Wastewater emissions": "Lượng xả thải nước sinh hoạt / công nghiệp",
    "COD emissions": "Phát thải COD",
    "SO2 emissions": "Phát thải SO2",
    "Soot and dust emissions": "Phát thải muội / bụi",
    "Industrial solid waste emissions": "Phát thải chất thải rắn công nghiệp",
    "Exhaust gas emission reduction treatment": "Xử lý / giảm phát thải khí thải",
    "Wastewater abatement treatment": "Xử lý / giảm thiểu nước thải",
    "Soot and dust management": "Quản lý muội / bụi",
    "Solid waste utilization and disposal": "Tận dụng và xử lý chất thải rắn",
    "whether it has a green invention patent": "Có bằng sáng chế phát minh xanh hay không",
    "whether it has a green utility model patent": "Có bằng sáng chế giải pháp hữu ích xanh hay không",
    "environmental violations/accidents/ petitions": "Vi phạm / tai nạn / khiếu kiện môi trường",
}

INSTRUCTION_VI = {
    "Disclosure: 2; Not disclosed: 0": "Có công bố: 2; Không công bố: 0",
    "Yes: 2; No: 0": "Có: 2; Không: 0",
    "No Description: 0; Qualitative: 1; Quantitative: 2": "Không mô tả: 0; Định tính: 1; Định lượng: 2",
    "Yes: -1; No: 0": "Có: -1; Không: 0",
}

SCORING_TYPE_VI = {
    "disclosure_binary": "Công bố / không công bố",
    "yes_no_binary": "Có / Không",
    "description_scale": "Không mô tả / định tính / định lượng",
    "negative_yes_penalty": "Có thì phạt / Không thì 0",
}

def vi_primary(text: str) -> str:
    return PRIMARY_VI.get(text, text)

def vi_secondary(text: str) -> str:
    return SECONDARY_VI.get(text, text)

def vi_instruction(text: str) -> str:
    return INSTRUCTION_VI.get(text, text)

def vi_scoring_type(text: str) -> str:
    return SCORING_TYPE_VI.get(text, text)


@dataclass
class Indicator:
    indicator_id: str
    primary_indicator: str
    secondary_indicator: str
    scoring_type: str
    scoring_instruction: str


def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_indicator_catalog() -> List[Indicator]:
    catalog: List[Indicator] = []
    for group in RUBRIC_GROUPS:
        primary = group["primary_indicator"]
        scoring_instruction = group["scoring_instruction"]
        scoring_type = group["scoring_type"]
        for secondary in group["secondary_indicators"]:
            indicator_id = f"{slugify(primary)}__{slugify(secondary)}"
            catalog.append(
                Indicator(
                    indicator_id=indicator_id,
                    primary_indicator=primary,
                    secondary_indicator=secondary,
                    scoring_type=scoring_type,
                    scoring_instruction=scoring_instruction,
                )
            )
    return catalog


INDICATOR_CATALOG = build_indicator_catalog()
INDICATOR_LOOKUP = {x.indicator_id: x for x in INDICATOR_CATALOG}
MAX_POSITIVE_SCORE = sum(max(SCORING_RULES[ind.scoring_type]["allowed_scores"]) for ind in INDICATOR_CATALOG)
MIN_POSSIBLE_SCORE = sum(min(SCORING_RULES[ind.scoring_type]["allowed_scores"]) for ind in INDICATOR_CATALOG)


class QuotaExceededError(RuntimeError):
    pass


class RateLimitExhaustedError(RuntimeError):
    pass


class LLMFormatError(RuntimeError):
    pass


def detect_tesseract_path() -> str:
    candidates = []

    env_path = os.getenv("TESSERACT_CMD", "").strip()
    if env_path:
        candidates.append(env_path)

    which_path = shutil.which("tesseract")
    if which_path:
        candidates.append(which_path)

    candidates.extend([
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
    ])

    seen = set()
    for candidate in candidates:
        candidate = str(candidate)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            return candidate

    raise RuntimeError(
        "Tesseract not found on Windows. Install Tesseract OCR, then either add it to PATH "
        "or set TESSERACT_CMD to something like: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )

TESSERACT_PATH = detect_tesseract_path()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_name(s: Any) -> str:
    s = "unknown" if s is None else str(s)
    s = re.sub(r"[^\w\-]+", "_", s)
    return s[:120]


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def score_label_vi(score: int, scoring_type: str) -> str:
    if scoring_type in {"disclosure_binary", "yes_no_binary"}:
        return "Có" if score == 2 else "Không"
    if scoring_type == "description_scale":
        return {0: "Không mô tả", 1: "Định tính", 2: "Định lượng"}.get(score, str(score))
    if scoring_type == "negative_yes_penalty":
        return "Có vi phạm" if score == -1 else "Không"
    return str(score)


def build_chunk_text_blob(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for idx, ch in enumerate(chunks, start=1):
        parts.append(f"[Chunk {idx}] chunk_id={ch['chunk_id']} score={ch['retrieval_score']:.4f}\n{ch['raw_text']}")
    return "\n\n==========\n\n".join(parts)


def validate_detail_row(row: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    allowed = SCORING_RULES[row['scoring_type']]['allowed_scores']
    score_final = row['score_final']
    evidence_chunk_ids = [x for x in str(row.get('evidence_chunk_ids', '')).split('|') if x != '']
    evidence_quote = str(row.get('evidence_quote_vi') or row.get('evidence_quote') or '').strip()
    conf = parse_float(row.get('confidence_0_1'), 0.0)
    grounding_conf = parse_float(row.get('grounding_confidence_0_1'), 0.0)
    top_retrieval = max([parse_float(ch.get('retrieval_score'), 0.0) for ch in retrieved_chunks], default=0.0)

    checks = {
        'validation_score_allowed': int(score_final in allowed),
        'validation_has_evidence_chunks': int(len(evidence_chunk_ids) > 0),
        'validation_has_evidence_quote': int(bool(evidence_quote)),
        'validation_top_retrieval_ge_0_02': int(top_retrieval >= 0.02),
        'validation_positive_score_has_support': int(not (score_final > 0 and not evidence_quote and len(evidence_chunk_ids) == 0)),
        'validation_negative_score_has_support': int(not (score_final < 0 and not evidence_quote and len(evidence_chunk_ids) == 0)),
    }
    hard_fail = checks['validation_score_allowed'] == 0
    soft_review = False
    if score_final != 0 and (checks['validation_has_evidence_quote'] == 0 or checks['validation_has_evidence_chunks'] == 0):
        soft_review = True
    if conf < 0.55 and row.get('judge_run', 0) == 0:
        soft_review = True
    if row.get('judge_run', 0) == 1 and grounding_conf not in (0.0, None) and grounding_conf < 0.55:
        soft_review = True
    if top_retrieval < 0.02 and score_final != 0:
        soft_review = True

    status = 'FAIL' if hard_fail else ('REVIEW' if soft_review else 'OK')
    notes = []
    if hard_fail:
        notes.append('Điểm cuối không nằm trong thang điểm hợp lệ')
    if score_final != 0 and not evidence_quote:
        notes.append('Điểm khác 0 nhưng thiếu trích dẫn bằng chứng')
    if score_final != 0 and len(evidence_chunk_ids) == 0:
        notes.append('Điểm khác 0 nhưng không có chunk hỗ trợ')
    if conf < 0.55 and row.get('judge_run', 0) == 0:
        notes.append('Độ tin cậy thấp nhưng chưa chạy judge')
    if row.get('judge_run', 0) == 1 and grounding_conf not in (0.0, None) and grounding_conf < 0.55:
        notes.append('Judge cho thấy độ bám bằng chứng thấp')
    if top_retrieval < 0.02 and score_final != 0:
        notes.append('Điểm retrieval thấp so với điểm chấm')

    return {
        **checks,
        'validation_evidence_chunk_count': len(evidence_chunk_ids),
        'validation_top_retrieval_score': round(top_retrieval, 6),
        'validation_status': status,
        'validation_notes_vi': ' | '.join(notes),
    }


def first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMFormatError(f"No JSON found in model output:\n{text}")
    return json.loads(text[start:end + 1])


def extract_message_content(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return ""


def wait_for_llm_slot():
    global LAST_LLM_CALL_TS
    elapsed = time.time() - LAST_LLM_CALL_TS
    if elapsed < MIN_SECONDS_BETWEEN_LLM_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_LLM_CALLS - elapsed)


def call_llm(model: str, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
    global LAST_LLM_CALL_TS

    if not LLM_API_KEY:
        raise RuntimeError("Missing LLM_API_KEY.")

    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    use_response_format = True

    for attempt in range(MAX_RETRIES):
        wait_for_llm_slot()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": max_tokens,
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            LAST_LLM_CALL_TS = time.time()

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                try:
                    body = resp.json()
                except Exception:
                    body = {}
                err_type = ((body.get("error") or {}).get("type") or "").lower()
                err_code = ((body.get("error") or {}).get("code") or "").lower()
                if err_type == "insufficient_quota" or err_code == "insufficient_quota":
                    raise QuotaExceededError("OpenAI quota exceeded.")
                time.sleep(min(30.0, 2 ** attempt))
                continue

            try:
                body = resp.json()
            except Exception:
                body = {}

            err_param = ((body.get("error") or {}).get("param") or "").strip()
            if resp.status_code == 400 and err_param == "response_format":
                use_response_format = False
                continue

            resp.raise_for_status()

        except requests.exceptions.RequestException:
            time.sleep(min(20.0, 2 ** attempt))

    raise RateLimitExhaustedError("LLM call failed after retries.")


def load_dataset() -> pd.DataFrame:
    source = DATASET_LOCAL_FILE if DATASET_LOCAL_FILE else DATASET_URL

    try:
        source_str = str(source).lower()

        if source_str.endswith(".xlsx") or source_str.endswith(".xls"):
            df = pd.read_excel(source)
        else:
            df = pd.read_csv(
                source,
                engine="python",
                quotechar='"',
                on_bad_lines="skip",
            )

    except Exception as e:
        raise RuntimeError(f"Could not read dataset source: {source}\n{e}")

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["ID", "year", "annual_report", "sustainability_report"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required dataset columns: {missing}. Found columns: {list(df.columns)}")

    return df.iloc[START_ROW - 1:END_ROW].copy()


def normalize_report_link(link: Any) -> Optional[str]:
    if pd.isna(link):
        return None
    link = str(link).strip()
    if not link or link == "#ERROR!":
        return None
    return link


def pick_report_link(row) -> Tuple[Optional[str], Optional[str]]:
    sustainability_link = normalize_report_link(getattr(row, "sustainability_report", None))
    annual_link = normalize_report_link(getattr(row, "annual_report", None))

    if sustainability_link:
        return sustainability_link, "sustainability_report"
    if annual_link:
        return annual_link, "annual_report"
    return None, None


def drive_to_direct(url: Any) -> Optional[str]:
    if pd.isna(url):
        return None
    url = str(url).strip()
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    if "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


def download_pdf(url: Optional[str], tries: int = 3) -> Optional[bytes]:
    if not url:
        return None
    headers = {"User-Agent": "Mozilla/5.0"}
    for _ in range(tries):
        try:
            r = session.get(url, headers=headers, timeout=120, allow_redirects=True)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return r.content
        except Exception:
            pass
        time.sleep(1.5)
    return None


def extract_pages_from_pdf(pdf_bytes: bytes, dpi: int = 300) -> Tuple[List[Dict[str, Any]], int, int]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Dict[str, Any]] = []
    total_pages = len(doc)
    ocr_pages_used = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        native_text = page.get_text("text").strip()
        use_ocr = OCR_ALL_PAGES or len(native_text.split()) < MIN_NATIVE_WORDS_PER_PAGE

        if use_ocr:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.SHARPEN)
            try:
                text = pytesseract.image_to_string(img, lang="vie+eng", config=OCR_CONFIG)
            except Exception:
                text = pytesseract.image_to_string(img, lang="eng", config=OCR_CONFIG)
            ocr_pages_used += 1
        else:
            text = native_text

        pages.append({
            "page_num": page_idx + 1,
            "text": text.strip(),
        })

    return pages, ocr_pages_used, total_pages


def cache_path_for_report(report_id: Any, year: Any) -> Path:
    return CACHE_DIR / f"{safe_name(report_id)}_{safe_name(year)}.json"


def load_or_extract_report_pages(report_id: Any, year: Any, pdf_url: Any) -> Tuple[List[Dict[str, Any]], int, int, int, str]:
    cache_path = cache_path_for_report(report_id, year)
    direct = drive_to_direct(pdf_url)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        return pages, 1, int(payload.get("used_ocr", 1)), -1, direct

    pdf_bytes = download_pdf(direct)
    if pdf_bytes is None:
        raise ValueError("Could not download valid PDF.")

    pages, ocr_pages_used, _ = extract_pages_from_pdf(pdf_bytes, dpi=OCR_DPI)
    payload = {"pages": pages, "used_ocr": int(ocr_pages_used > 0)}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return pages, 0, int(ocr_pages_used > 0), ocr_pages_used, direct


def chunk_pages(pages: List[Dict[str, Any]], chunk_words: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    tokens: List[Tuple[int, str]] = []
    for p in pages:
        page_num = p["page_num"]
        for w in str(p.get("text", "")).split():
            tokens.append((page_num, w))

    if not tokens:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0
    while start < len(tokens):
        end = min(start + chunk_words, len(tokens))
        window = tokens[start:end]
        words = [w for _, w in window]
        page_nums = [p for p, _ in window]
        raw_text = " ".join(words).strip()
        if raw_text:
            chunks.append({
                "chunk_id": chunk_id,
                "page_start": min(page_nums),
                "page_end": max(page_nums),
                "word_count": len(words),
                "raw_text": raw_text[:MAX_CHUNK_CHARS_IN_PROMPT],
                "clean_text": clean_text(raw_text),
            })
            chunk_id += 1
        if end == len(tokens):
            break
        start = max(end - overlap, start + 1)
    return chunks


def indicator_queries(indicator: Indicator) -> List[str]:
    hints = INDICATOR_HINTS.get(indicator.secondary_indicator, [])
    secondary_vi = vi_secondary(indicator.secondary_indicator)
    primary_vi = vi_primary(indicator.primary_indicator)
    queries = [
        indicator.secondary_indicator,
        secondary_vi,
        f"{indicator.primary_indicator} {indicator.secondary_indicator}",
        f"{primary_vi} {secondary_vi}",
    ]
    if hints:
        queries.append(" ".join(hints[:6]))
    if indicator.scoring_type == "description_scale":
        queries.append(f"{indicator.secondary_indicator} {secondary_vi} metric amount volume total percentage treatment xử lý định lượng số liệu")
    elif indicator.scoring_type == "yes_no_binary":
        queries.append(f"{indicator.secondary_indicator} {secondary_vi} certified approved compliant disclosed yes có chứng nhận có công bố")
    elif indicator.scoring_type == "disclosure_binary":
        queries.append(f"{indicator.secondary_indicator} {secondary_vi} disclose described policy report công bố báo cáo mô tả")
    elif indicator.scoring_type == "negative_yes_penalty":
        queries.append(f"{indicator.secondary_indicator} {secondary_vi} violation fine penalty accident petition sanction vi phạm xử phạt tai nạn khiếu kiện")
    return [clean_text(x) for x in queries if clean_text(x)]


def build_vectorizer_and_matrix(chunks: List[Dict[str, Any]]):
    texts = [c["clean_text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_indicator_chunks(
    chunks: List[Dict[str, Any]],
    indicator: Indicator,
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    scores = [0.0 for _ in chunks]
    for q in indicator_queries(indicator):
        qv = vectorizer.transform([clean_text(q)])
        sims = cosine_similarity(qv, matrix).flatten()
        scores = [max(scores[i], float(sims[i])) for i in range(len(scores))]

    for i, ch in enumerate(chunks):
        text_low = ch["raw_text"].lower()
        if indicator.scoring_type == "description_scale":
            if re.search(r"\b\d+(?:[.,]\d+)?\b", text_low):
                scores[i] += 0.015
            if re.search(r"\b(?:%|percent|tấn|tons?|tonnes?|kg|m3|m³|kwh|mwh|mw|co2|tco2e|cod|so2)\b", text_low):
                scores[i] += 0.02
        if indicator.scoring_type == "negative_yes_penalty":
            if re.search(r"\b(?:violation|violations|accident|accidents|petition|petitions|fine|penalty|complaint|breach|xử phạt|vi phạm|tai nạn|khiếu kiện)\b", text_low):
                scores[i] += 0.03

    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    out = []
    for idx in idxs:
        ch = dict(chunks[idx])
        ch["retrieval_score"] = round(float(scores[idx]), 8)
        out.append(ch)
    return out


def build_batch_context(indicators: List[Indicator], retrieved: Dict[str, List[Dict[str, Any]]]) -> str:
    used_chunks = {}
    for ind in indicators:
        for ch in retrieved.get(ind.indicator_id, []):
            key = (ch["chunk_id"], ch.get("page_start"), ch.get("page_end"), ch["raw_text"])
            used_chunks[key] = ch["retrieval_score"]

    ranked = sorted(used_chunks.items(), key=lambda x: x[1], reverse=True)[:MAX_TOTAL_CHUNKS_PER_BATCH]
    blocks = []
    for i, ((chunk_id, page_start, page_end, raw_text), score) in enumerate(ranked, start=1):
        blocks.append(f"[Evidence {i}] chunk_id={chunk_id} pages={page_start}-{page_end} retrieval_score={score:.4f}\n{raw_text}")
    return "\n\n".join(blocks)


def build_indicator_list_text(indicators: List[Indicator]) -> str:
    lines = []
    for ind in indicators:
        lines.append(
            f"- indicator_id={ind.indicator_id} | primary={ind.primary_indicator} | primary_vi={vi_primary(ind.primary_indicator)} | secondary={ind.secondary_indicator} | secondary_vi={vi_secondary(ind.secondary_indicator)} | scoring_type={ind.scoring_type} | scoring_instruction={ind.scoring_instruction}"
        )
    return "\n".join(lines)

def score_batch(indicators: List[Indicator], retrieved: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not indicators:
        return {}

    context = build_batch_context(indicators, retrieved)
    indicator_list = build_indicator_list_text(indicators)
    scoring_type = indicators[0].scoring_type

    messages = [
        {
            "role": "system",
            "content": (
    "You are a careful rubric grader for annual reports and sustainability reports. "
    "Use only the provided report evidence chunks to score each indicator. "
    "Do not use outside knowledge or make unsupported assumptions. "
    "Score each indicator strictly according to its scoring_instruction and allowed score scale. "
    "If the evidence is unclear or insufficient, choose the lower justified score. "
    "For any non-zero or penalty score, provide direct supporting evidence from the chunks. "
    "Do not treat general ESG language or vague statements as strong evidence unless they clearly match the indicator. "
    "For description_scale indicators, give 2 only when the evidence is clearly quantitative or measurable; give 1 for clear qualitative description; give 0 if not described. "
    "For binary indicators, give the positive score only when the condition is explicitly supported by the evidence. "
    "For negative_yes_penalty indicators, assign the negative score only when a violation, accident, petition, penalty, or similar issue is explicitly stated. "
    "Return valid JSON only, with no markdown or extra commentary. "
    "Follow the exact requested schema. "
    "Write reason in Vietnamese. "
    "Write evidence_quote in Vietnamese when possible, or keep the original English wording if the source evidence is in English. "
    "Use only chunk IDs that appear in the provided evidence. "
    "If evidence is weak, borderline, or indirect, set flag_for_judge to true."
),
        },
        {
            "role": "user",
            "content": (
                "Score these indicators in ONE batch.\n\n"
                f"Common scoring_type for this batch: {scoring_type}\n\n"
                "Indicators:\n"
                f"{indicator_list}\n\n"
                "Return JSON with this exact shape:\n"
                "{\n"
                '  "results": [\n'
                "    {\n"
                '      "indicator_id": "...",\n'
                '      "score": 0,\n'
                '      "confidence_0_1": 0.0,\n'
                '      "reason": "brief reason",\n'
                '      "evidence_chunk_ids": [1,2],\n'
                '      "evidence_quote": "short quote or paraphrase",\n'
                '      "flag_for_judge": false\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                "- disclosure_binary: allowed scores are 0 or 2 only\n"
                "- yes_no_binary: allowed scores are 0 or 2 only\n"
                "- description_scale: allowed scores are 0, 1, or 2 only\n"
                "- negative_yes_penalty: allowed scores are -1 or 0 only\n"
                "- Use 2 only when evidence clearly supports the positive or quantitative case under the exact rubric.\n"
                "- For description_scale, use 2 only when the evidence is indicator-specific and quantitative. Generic ESG language is not enough.\n"
                "- Use 1 only for qualitative description under description_scale.\n"
                "- Use 0 when evidence is absent, insufficient, or only tangential.\n"
                "- flag_for_judge should be true when confidence is low, evidence is thin, or the score is unusually strong/negative.\n\n"
                f"Evidence:\n{context}"
            ),
        },
    ]

    data = call_llm(LLM_MODEL_SCORER, messages, max_tokens=LLM_MAX_TOKENS_SCORE_BATCH)
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = extract_message_content(message)
    if not content:
        raise LLMFormatError("Empty scorer content")
    parsed = first_json_object(content)
    usage = data.get("usage", {})
    parsed["usage"] = usage
    return parsed


def judge_one_indicator(indicator: Indicator, candidate: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    context_blocks = []
    for i, ch in enumerate(retrieved_chunks[:TOP_K_PER_INDICATOR], start=1):
        context_blocks.append(f"[Chunk {i}] chunk_id={ch['chunk_id']} pages={ch.get('page_start')}-{ch.get('page_end')} retrieval_score={ch['retrieval_score']:.4f}\n{ch['raw_text']}")
    context = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grounding judge. Check whether the proposed score is justified by the evidence. "
                "Be conservative. Return JSON only. Write judge_reason in Vietnamese."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return JSON with this exact shape:\n"
                "{\n"
                '  "verdict": "KEEP|LOWER|RAISE",\n'
                '  "final_score": 0,\n'
                '  "judge_reason": "brief reason",\n'
                '  "grounding_confidence_0_1": 0.0\n'
                "}\n\n"
                f"Indicator: {indicator.secondary_indicator}\n"
                f"Primary: {indicator.primary_indicator}\n"
                f"Scoring instruction: {indicator.scoring_instruction}\n"
                f"Allowed scores: {SCORING_RULES[indicator.scoring_type]['allowed_scores']}\n\n"
                f"Proposed result:\n{json.dumps(candidate, ensure_ascii=False)}\n\n"
                f"Evidence:\n{context}"
            ),
        },
    ]

    data = call_llm(LLM_MODEL_JUDGE, messages, max_tokens=LLM_MAX_TOKENS_JUDGE)
    choice = data["choices"][0]
    message = choice.get("message", {})
    content = extract_message_content(message)
    if not content:
        raise LLMFormatError("Empty judge content")
    return first_json_object(content)


def select_items_for_judge(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flagged = []
    for r in rows:
        score = r.get("score")
        conf = parse_float(r.get("confidence_0_1"), 0.0)
        if r.get("flag_for_judge"):
            flagged.append(r)
            continue
        if conf < 0.65:
            flagged.append(r)
            continue
        if score == -1 or score == 2:
            flagged.append(r)
    # prioritize lowest confidence first
    flagged = sorted(flagged, key=lambda x: parse_float(x.get("confidence_0_1"), 0.0))
    # dedupe by indicator_id
    seen = set()
    out = []
    for row in flagged:
        iid = row["indicator_id"]
        if iid not in seen:
            out.append(row)
            seen.add(iid)
        if len(out) >= MAX_JUDGED_INDICATORS_PER_REPORT:
            break
    return out


def merge_old_and_new(path: Path, new_df: pd.DataFrame, subset_cols: List[str]) -> pd.DataFrame:
    if path.exists():
        try:
            old_df = pd.read_csv(path)
            out = pd.concat([old_df, new_df], ignore_index=True)
            out = out.drop_duplicates(subset=subset_cols, keep="last")
            return out
        except Exception:
            return new_df
    return new_df

def merge_from_many(paths: List[Path], new_df: pd.DataFrame, subset_cols: List[str]) -> pd.DataFrame:
    out = new_df.copy()
    for path in paths:
        out = merge_old_and_new(path, out, subset_cols)
    return out

def process_one_report(row) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    report_id = getattr(row, "ID", None)
    year = getattr(row, "year", None)
    link, report_source = pick_report_link(row)

    if not link:
        raise ValueError("Empty sustainability_report and annual_report links.")

    pages, used_cache, used_ocr, ocr_pages_used, direct_pdf_url = load_or_extract_report_pages(report_id, year, link)
    raw_text = "\n\n".join(str(p.get("text", "")) for p in pages)
    if len(raw_text.split()) < 80:
        raise ValueError("Too little extracted text.")

    chunks = chunk_pages(pages, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("No chunks generated.")

    vectorizer, matrix = build_vectorizer_and_matrix(chunks)

    # local retrieval once per indicator (cheap, no LLM)
    retrieved: Dict[str, List[Dict[str, Any]]] = {}
    for ind in INDICATOR_CATALOG:
        retrieved[ind.indicator_id] = retrieve_indicator_chunks(chunks, ind, vectorizer, matrix, top_k=TOP_K_PER_INDICATOR)

    # batch score by scoring type => 4 scorer calls max per report
    detail_rows: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_tokens = 0

    for scoring_type in ["disclosure_binary", "yes_no_binary", "description_scale", "negative_yes_penalty"]:
        inds = [x for x in INDICATOR_CATALOG if x.scoring_type == scoring_type]
        batch_result = score_batch(inds, retrieved)
        usage = batch_result.get("usage", {})
        total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        total_api_tokens += int(usage.get("total_tokens", 0) or 0)

        result_map = {x["indicator_id"]: x for x in batch_result.get("results", [])}
        for ind in inds:
            candidate = result_map.get(ind.indicator_id, {})
            score = candidate.get("score", 0)
            allowed = SCORING_RULES[ind.scoring_type]["allowed_scores"]
            if score not in allowed:
                score = min(allowed, key=lambda z: abs(z - parse_float(score, 0)))

            retrieved_chunks = retrieved[ind.indicator_id]
            detail_rows.append({
                "ID": report_id,
                "year": year,
                "primary_indicator": ind.primary_indicator,
                "primary_indicator_vi": vi_primary(ind.primary_indicator),
                "secondary_indicator": ind.secondary_indicator,
                "secondary_indicator_vi": vi_secondary(ind.secondary_indicator),
                "indicator_id": ind.indicator_id,
                "scoring_type": ind.scoring_type,
                "scoring_type_vi": vi_scoring_type(ind.scoring_type),
                "scoring_instruction": ind.scoring_instruction,
                "scoring_instruction_vi": vi_instruction(ind.scoring_instruction),
                "score_initial": score,
                "score_final": score,
                "score_label_initial_vi": score_label_vi(score, ind.scoring_type),
                "score_label_final_vi": score_label_vi(score, ind.scoring_type),
                "confidence_0_1": parse_float(candidate.get("confidence_0_1"), 0.0),
                "reason_vi": str(candidate.get("reason", "")).strip(),
                "reason": str(candidate.get("reason", "")).strip(),
                "evidence_chunk_ids": "|".join(map(str, candidate.get("evidence_chunk_ids", []))),
                "evidence_quote_vi": str(candidate.get("evidence_quote", "")).strip(),
                "evidence_quote": str(candidate.get("evidence_quote", "")).strip(),
                "page_range": "|".join(sorted({f"{ch.get('page_start')}-{ch.get('page_end')}" for ch in retrieved_chunks if ch.get('chunk_id') in set(candidate.get('evidence_chunk_ids', []))})),
                "retrieved_chunk_ids": "|".join(str(x["chunk_id"]) for x in retrieved_chunks),
                "retrieved_chunk_scores": "|".join(f"{x['retrieval_score']:.6f}" for x in retrieved_chunks),
                "best_retrieved_chunk_text": retrieved_chunks[0]["raw_text"] if retrieved_chunks else "",
                "retrieved_chunk_texts": build_chunk_text_blob(retrieved_chunks),
                "judge_run": 0,
                "judge_verdict": None,
                "judge_reason_vi": None,
                "judge_reason": None,
                "grounding_confidence_0_1": None,
                "report_source": report_source,
                "annual_report_link": normalize_report_link(getattr(row, "annual_report", None)),
                "sustainability_report_link": normalize_report_link(getattr(row, "sustainability_report", None)),
                "selected_report_link": link,
                "pdf_link": direct_pdf_url,
            })

    # selective judge
    if RUN_JUDGE:
        to_judge = select_items_for_judge(detail_rows)
        for row_detail in detail_rows:
            if row_detail["indicator_id"] not in {x["indicator_id"] for x in to_judge}:
                continue
            ind = INDICATOR_LOOKUP[row_detail["indicator_id"]]
            candidate = {
                "score": row_detail["score_initial"],
                "confidence_0_1": row_detail["confidence_0_1"],
                "reason": row_detail["reason"],
                "evidence_quote": row_detail["evidence_quote"],
            }
            judge = judge_one_indicator(ind, candidate, retrieved[ind.indicator_id])
            allowed = SCORING_RULES[ind.scoring_type]["allowed_scores"]
            final_score = judge.get("final_score", row_detail["score_initial"])
            if final_score not in allowed:
                final_score = row_detail["score_initial"]
            row_detail["judge_run"] = 1
            row_detail["judge_verdict"] = str(judge.get("verdict", "")).strip()
            row_detail["judge_reason_vi"] = str(judge.get("judge_reason", "")).strip()
            row_detail["judge_reason"] = str(judge.get("judge_reason", "")).strip()
            row_detail["grounding_confidence_0_1"] = parse_float(judge.get("grounding_confidence_0_1"), 0.0)
            row_detail["score_final"] = final_score
            row_detail["score_label_final_vi"] = score_label_vi(final_score, ind.scoring_type)

    for row_detail in detail_rows:
        row_detail.update(validate_detail_row(row_detail, retrieved.get(row_detail["indicator_id"], [])))

    details_df = pd.DataFrame(detail_rows)
    subtotal_df = details_df.groupby("primary_indicator", as_index=False)["score_final"].sum()
    subtotal_map = {r["primary_indicator"]: r["score_final"] for _, r in subtotal_df.iterrows()}

    record = {
        "ID": report_id,
        "year": year,
        "report_source": report_source,
        "annual_report_link": normalize_report_link(getattr(row, "annual_report", None)),
        "sustainability_report_link": normalize_report_link(getattr(row, "sustainability_report", None)),
        "selected_report_link": link,
        "pdf_link": direct_pdf_url,
        "used_cache": used_cache,
        "used_ocr": used_ocr,
        "ocr_pages_used": ocr_pages_used,
        "word_count": len(raw_text.split()),
        "n_pages": len(pages),
        "n_chunks_total": len(chunks),
        "tong_diem_cuoi": float(details_df["score_final"].sum()),
        "total_score_final": float(details_df["score_final"].sum()),
        "max_positive_score": MAX_POSITIVE_SCORE,
        "min_possible_score": MIN_POSSIBLE_SCORE,
        "judge_count": int(details_df["judge_run"].sum()),
        "so_chi_tieu_review": int((details_df["validation_status"] == "REVIEW").sum()),
        "so_chi_tieu_fail": int((details_df["validation_status"] == "FAIL").sum()),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens_api": total_api_tokens,
    }

    for primary in subtotal_map:
        record[f"subtotal__{slugify(primary)}"] = subtotal_map[primary]
        record[f"tong_phu__{slugify(vi_primary(primary))}"] = subtotal_map[primary]

    return record, detail_rows


def main():
    print("Tesseract path:", TESSERACT_PATH)
    print("Cache dir:", CACHE_DIR)
    print("LLM base URL:", LLM_BASE_URL)
    print("LLM_MODEL_SCORER:", LLM_MODEL_SCORER)
    print("LLM_MODEL_JUDGE:", LLM_MODEL_JUDGE)
    print("RUN_JUDGE:", RUN_JUDGE)
    print("OCR_ALL_PAGES:", OCR_ALL_PAGES)
    print("Rows:", START_ROW, "to", END_ROW)
    print("Rubric indicators:", len(INDICATOR_CATALOG))
    print("Score range:", MIN_POSSIBLE_SCORE, "to", MAX_POSITIVE_SCORE)

    df_data = load_dataset()
    results = []
    details = []
    errors = []
    new_count = 0

    for i, row in enumerate(df_data.itertuples(index=False), start=1):
        report_id = getattr(row, "ID", None)
        year = getattr(row, "year", None)
        try:
            record, detail_rows = process_one_report(row)
            results.append(record)
            details.extend(detail_rows)
            new_count += 1
            print(
                f"Processed {i} | ID={report_id} | year={year} | total={record['total_score_final']} | judges={record['judge_count']}"
            )
            if new_count % BATCH_SAVE_EVERY == 0:
                partial_results_df = merge_from_many(
                    [PARTIAL_OUTPUT_PATH, OUTPUT_CSV_PATH],
                    pd.DataFrame(results),
                    ["ID", "year"],
                )
                partial_details_df = merge_from_many(
                    [PARTIAL_DETAIL_PATH, DETAIL_CSV_PATH],
                    pd.DataFrame(details),
                    ["ID", "year", "indicator_id"],
                )
                partial_errors_df = merge_from_many(
                    [PARTIAL_ERROR_PATH, ERROR_CSV_PATH],
                    pd.DataFrame(errors),
                    ["ID", "year", "error"],
                )

                partial_results_df.to_csv(PARTIAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
                partial_details_df.to_csv(PARTIAL_DETAIL_PATH, index=False, encoding="utf-8-sig")
                partial_errors_df.to_csv(PARTIAL_ERROR_PATH, index=False, encoding="utf-8-sig")

                print("Saved partial outputs safely.")


        except Exception as e:
            selected_link, report_source = pick_report_link(row)
            errors.append({
                "ID": report_id,
                "year": year,
                "error": str(e),
                "report_source": report_source,
                "annual_report_link": normalize_report_link(getattr(row, "annual_report", None)),
                "sustainability_report_link": normalize_report_link(getattr(row, "sustainability_report", None)),
                "selected_report_link": selected_link,
            })
            print(f"Error on ID={report_id}, year={year}: {e}")

    results_df = merge_from_many(
        [PARTIAL_OUTPUT_PATH, OUTPUT_CSV_PATH],
        pd.DataFrame(results),
        ["ID", "year"],
    )

    details_df = merge_from_many(
        [PARTIAL_DETAIL_PATH, DETAIL_CSV_PATH],
        pd.DataFrame(details),
        ["ID", "year", "indicator_id"],
    )

    errors_df = merge_from_many(
        [PARTIAL_ERROR_PATH, ERROR_CSV_PATH],
        pd.DataFrame(errors),
        ["ID", "year", "error"],
    )

    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    results_df.to_excel(OUTPUT_XLSX_PATH, index=False)
    details_df.to_csv(DETAIL_CSV_PATH, index=False, encoding="utf-8-sig")
    details_df.to_excel(DETAIL_XLSX_PATH, index=False)
    errors_df.to_csv(ERROR_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Saved:", OUTPUT_CSV_PATH.name)
    print("Saved:", OUTPUT_XLSX_PATH.name)
    print("Saved:", DETAIL_CSV_PATH.name)
    print("Saved:", DETAIL_XLSX_PATH.name)
    print("Saved:", ERROR_CSV_PATH.name)
    print("Vietnamese output + validation + chunk text added.")


if __name__ == "__main__":
    main()
