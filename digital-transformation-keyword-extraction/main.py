# 1. SILENCE BLOCK (MUST BE FIRST)
import warnings
import os
import logging
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.disable(logging.CRITICAL)

import io
import json
import re
import sys
import time
import requests
import fitz
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import ahocorasick

# Silence fitz (MuPDF) native errors
fitz.TOOLS.mupdf_display_errors(False)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# =========================
# PATHS / SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent
KEYWORD_PATH = BASE_DIR / "keyword_dictionary-ficc.csv"

OUTPUT_CSV_PATH = BASE_DIR / "dt_results.csv"
OUTPUT_XLSX_PATH = BASE_DIR / "dt_results.xlsx"
VALIDATION_CSV_PATH = BASE_DIR / "dt_validation.csv"
VALIDATION_XLSX_PATH = BASE_DIR / "dt_validation.xlsx"
PARTIAL_OUTPUT_PATH = BASE_DIR / "dt_results_partial.csv"
PARTIAL_VALIDATION_PATH = BASE_DIR / "dt_validation_partial.csv" 
PROGRESS_FILE = BASE_DIR / "progress.json"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

DATASET_URL = os.getenv("DATASET_URL")
if not DATASET_URL:
    raise ValueError("Please set DATASET_URL as an environment variable.")
START_ROW = 1
END_ROW = 1000000
DPI = 300  
MIN_CHARS_PER_PAGE = 300
LOG_BATCH = 100
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

TESSDATA_DIR = os.getenv("TESSDATA_DIR")
if TESSDATA_DIR:
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# Pre-compile expensive text cleaning regex (used per-file)
_RE_CLEAN_VI = re.compile(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_WORD_CHAR = re.compile(r"\w")  # for word-boundary check in Aho-Corasick

# =========================
# GLOBAL DATA LOAD
# =========================
def load_keywords():
    try:
        df_kw = pd.read_csv(KEYWORD_PATH, sep=",", encoding="utf-8-sig", engine="python")
        df_kw.columns = df_kw.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        df_kw = df_kw.dropna(axis=1, how="all")

        # Validate required columns
        required_cols = ["Dimension", "Keyword_vi", "Keyword_en"]
        missing_cols = [col for col in required_cols if col not in df_kw.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}. Available: {df_kw.columns.tolist()}")

        df_kw["Dimension"] = df_kw["Dimension"].fillna("").astype(str).str.strip()
        df_kw["Keyword_vi"] = df_kw["Keyword_vi"].fillna("").astype(str).str.strip()
        df_kw["Keyword_en"] = df_kw["Keyword_en"].fillna("").astype(str).str.strip()

        norm_vi = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", str(s).lower())).strip()
        norm_en = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()

        df_kw["Keyword_vi_clean"] = df_kw["Keyword_vi"].apply(norm_vi)
        df_kw["Keyword_en_clean"] = df_kw["Keyword_en"].apply(norm_en)
        df_kw = df_kw[(df_kw["Keyword_vi_clean"] != "") | (df_kw["Keyword_en_clean"] != "")].copy()

        # FIX: Deduplicate keywords (matching main.py)
        df_kw = df_kw.drop_duplicates(subset=["Dimension", "Keyword_vi_clean", "Keyword_en_clean"])

        kw_vi = df_kw.groupby("Dimension")["Keyword_vi_clean"].apply(lambda s: [x for x in s if x]).to_dict()
        kw_en = df_kw.groupby("Dimension")["Keyword_en_clean"].apply(lambda s: [x for x in s if x]).to_dict()

        # OPTIMIZATION: Build Aho-Corasick automaton for single-pass multi-pattern matching
        # Instead of scanning text N times (once per keyword), scan ONCE for ALL keywords
        automaton = ahocorasick.Automaton()
        all_dims = set(kw_vi.keys()) | set(kw_en.keys())
        for dim in all_dims:
            for phrase in kw_vi.get(dim, []):
                automaton.add_word(phrase, (dim, phrase, "vi"))
            for phrase in kw_en.get(dim, []):
                automaton.add_word(phrase, (dim, phrase, "en"))
        automaton.make_automaton()

        return kw_vi, kw_en, automaton, all_dims
    except Exception as e:
        print(f"Error loading keywords: {e}")
        sys.exit(1)

GLOBAL_KEYWORDS_VI, GLOBAL_KEYWORDS_EN, AC_AUTOMATON, ALL_DIMS = load_keywords()

def _is_word_boundary(text, start, end):
    """Check if match at [start:end] has word boundaries on both sides."""
    if start > 0 and _RE_WORD_CHAR.match(text[start - 1]):
        return False
    if end < len(text) and _RE_WORD_CHAR.match(text[end]):
        return False
    return True

# =========================
# LOG SYSTEM
# =========================
log_buffer = []

def log(msg=""):
    print(msg)
    log_buffer.append(msg)

def save_log_file(start_row, end_row):
    if not log_buffer:
        return
    filename = LOG_DIR / f"log_parallel_{start_row}-{end_row}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Parallel Processing Log: Rows {start_row} - {end_row}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n")
        for line in log_buffer:
            f.write(line + "\n")
        f.write("```\n")
    log_buffer.clear()
    print(f"  [LOG] Saved {filename.name}")

# =========================
# WORKER FUNCTIONS
# =========================

# Per-worker session (connection pooling within each process)
_worker_session = None

def _get_session():
    global _worker_session
    if _worker_session is None:
        _worker_session = requests.Session()
        _worker_session.headers.update({"User-Agent": "Mozilla/5.0"})
    return _worker_session

def drive_to_direct(url):
    if pd.isna(url): return None
    url = str(url).strip()
    if "/file/d/" in url:
        return f"https://drive.google.com/uc?export=download&id={url.split('/file/d/')[1].split('/')[0]}"
    # FIX: Handle id= URL format (matching main.py)
    if "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def download_pdf(url, tries=3, sleep_sec=2):
    """FIX: Retry download 3 times with sleep (matching main.py)."""
    if url is None:
        return None
    session = _get_session()
    for _ in range(tries):
        try:
            r = session.get(url, timeout=120, allow_redirects=True)
            if r.status_code == 200 and (r.content[:4] == b"%PDF" or "application/pdf" in r.headers.get("Content-Type", "")):
                return r.content
        except Exception:
            pass
        time.sleep(sleep_sec)
    return None

def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text, ocr_pages = [], 0
    config = f'--oem 1 --psm 3'
    try:
        for page in doc:
            native = (page.get_text("text") or "").strip()
            if len(native) >= MIN_CHARS_PER_PAGE:
                all_text.append(native)
                continue
            ocr_pages += 1
            mat = fitz.Matrix(DPI/72, DPI/72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.SHARPEN)
            try:
                txt = pytesseract.image_to_string(img, lang="vie+eng", config=config)
            except Exception:
                txt = pytesseract.image_to_string(img, lang="eng", config=config)
            all_text.append(txt)
    finally:
        doc.close()
    return "\n".join(all_text).strip(), ocr_pages

def process_single_row(row_data):
    """Process one PDF: download, extract text, count keywords."""
    # Silence warnings in worker processes
    warnings.filterwarnings("ignore")
    fitz.TOOLS.mupdf_display_errors(False)
    if TESSDATA_DIR:
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

    idx = row_data['idx']
    row_id = row_data['ID']
    link = row_data['annual_report']
    company_name = row_data['company_name']
    ma_ck = row_data['ma_ck']
    year = row_data['year']

    if pd.isna(link) or not link:
        return idx, "SKIP", None, [], None

    direct = drive_to_direct(link)
    pdf_bytes = download_pdf(direct)
    if pdf_bytes is None:
        return idx, "FAIL_PDF", None, [], None

    try:
        raw_text, ocr_pages = extract_text(pdf_bytes)
        del pdf_bytes  # Free memory early

        # OPTIMIZATION: Use pre-compiled regex instead of re.sub() each time
        clean_vi = _RE_WHITESPACE.sub(" ", _RE_CLEAN_VI.sub(" ", str(raw_text).lower())).strip()
        word_count = len(clean_vi.split())
        used_ocr = int(ocr_pages > 0)

        record = {
            "ID": row_id, "year": year,
            "annual_report_link": link, "pdf_link": direct,
            "used_ocr": used_ocr, "ocr_pages_used": ocr_pages,
            "word_count": word_count, "ma_ck": ma_ck, "company_name": company_name
        }

        v_rows = []
        total_vi, total_en = 0, 0
        dim_hits_summary = {}

        # OPTIMIZATION: Aho-Corasick single-pass scan — O(text_length + matches)
        # Collect all matches in one pass, grouped by (dim, phrase, lang)
        raw_matches = {}  # (dim, phrase, lang) -> list of (start, end)
        text_len = len(clean_vi)
        for end_pos, (dim, phrase, lang) in AC_AUTOMATON.iter(clean_vi):
            start_pos = end_pos - len(phrase) + 1
            # Post-filter: check word boundaries (Aho-Corasick doesn't support \b)
            if _is_word_boundary(clean_vi, start_pos, end_pos + 1):
                key = (dim, phrase, lang)
                if key not in raw_matches:
                    raw_matches[key] = []
                raw_matches[key].append((start_pos, end_pos + 1))

        # Aggregate matches by dimension — BINARY counting (0/1 per keyword)
        dim_data = {}  # dim -> {d_vi, d_en, m_vi, m_en}
        for (dim, phrase, lang), positions in raw_matches.items():
            if dim not in dim_data:
                dim_data[dim] = {"d_vi": 0, "d_en": 0, "m_vi": [], "m_en": []}
            dd = dim_data[dim]
            # Binary: each keyword counts as 1, regardless of frequency
            if lang == "vi":
                dd["d_vi"] += 1
                dd["m_vi"].append(phrase)
            else:
                dd["d_en"] += 1
                dd["m_en"].append(phrase)
            # Keep only first snippet per keyword for validation
            first_start, first_end = positions[0]
            v_rows.append({
                "ID": row_id, "year": year, "dimension": dim,
                "keyword": phrase, "keyword_lang": lang,
                "count_for_keyword": 1,  # Binary
                "snippet": clean_vi[max(0, first_start - 80):min(text_len, first_end + 80)],
                "pdf_link": direct, "annual_report_link": link,
                "used_ocr": used_ocr,
            })

        # Build record fields for ALL dimensions (including those with 0 hits)
        for dim in ALL_DIMS:
            dd = dim_data.get(dim, {"d_vi": 0, "d_en": 0, "m_vi": [], "m_en": []})
            d_vi, d_en = dd["d_vi"], dd["d_en"]
            record.update({
                f"{dim}_count_vi": d_vi, f"{dim}_count_en": d_en,
                f"{dim}_count_total": d_vi + d_en,
                f"{dim}_keywords_vi": "|".join(dd["m_vi"]),
                f"{dim}_keywords_en": "|".join(dd["m_en"]),
                f"{dim}_rate_per_1000": (d_vi + d_en) / word_count * 1000 if word_count else 0,
            })
            total_vi += d_vi
            total_en += d_en
            if d_vi + d_en > 0:
                dim_hits_summary[dim] = {"total": d_vi + d_en, "kws": dd["m_vi"] + dd["m_en"]}

        record.update({"TOTAL_vi": total_vi, "TOTAL_en": total_en, "TOTAL": total_vi + total_en})
        return idx, "OK", record, v_rows, dim_hits_summary
    except Exception as e:
        return idx, f"ERROR: {str(e)}", None, [], None

# =========================
# SYSTEM CONTROLLER
# =========================
def run_parallel():
    processed_ids, results, validation_rows = set(), [], []

    # Resume system
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                prog = json.load(f)
                processed_ids = set(map(str, prog.get("processed_ids", [])))
        except Exception:
            pass

    if PARTIAL_OUTPUT_PATH.exists():
        try:
            df_partial = pd.read_csv(PARTIAL_OUTPUT_PATH, encoding="utf-8-sig")
            if not df_partial.empty:
                results = df_partial.to_dict("records")
                for r in results:
                    processed_ids.add(str(r.get("ID", "")))
                log(f"  [RESUME] Loaded {len(results)} previous results from partial CSV")
        except pd.errors.EmptyDataError:
            log("  [RESUME] Partial results file exists but is empty. Starting fresh.")
        except Exception as e:
            log(f"  [RESUME] Warning: Could not load partial results: {e}")

    if PARTIAL_VALIDATION_PATH.exists():
        try:
            df_v_partial = pd.read_csv(PARTIAL_VALIDATION_PATH, encoding="utf-8-sig")
            if not df_v_partial.empty:
                validation_rows = df_v_partial.to_dict("records")
                log(f"  [RESUME] Loaded {len(validation_rows)} previous validation rows")
        except pd.errors.EmptyDataError:
            pass
        except Exception as e:
            log(f"  [RESUME] Warning: Could not load partial validation data: {e}")
    
    df_data = pd.read_csv(DATASET_URL).iloc[START_ROW-1:END_ROW]
    tasks = []
    for i, row in enumerate(df_data.itertuples(index=False), start=1):
        row_id = str(getattr(row, "ID", "N/A"))
        if row_id in processed_ids:
            continue
        tasks.append({
            'idx': i, 'ID': row_id,
            'company_name': getattr(row, "company_name", "N/A"),
            'ma_ck': getattr(row, "ma_ck", "N/A"),
            'year': getattr(row, "year", "N/A"),
            'annual_report': getattr(row, "annual_report", None),
        })

    if not tasks: 
        log("Tất cả dữ liệu đã được xử lý xong.")
        return

    workers = MAX_WORKERS
    total_to_do = len(tasks)

    log("\n" + "=" * 80)
    log(f"  [PARALLEL] Workers: {workers}")    
    log(f"  Processing {total_to_do} files...")
    log(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80 + "\n")
    
    start_time_all = time.time()
    finished_count = 0
    ok_count, skip_count, fail_count, error_count = 0, 0, 0, 0
    log_batch_start = 1

    # Single executor for all tasks — avoid process spawn overhead per batch
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Throttled submission: keep at most workers*3 futures in-flight
        MAX_INFLIGHT = workers * 3
        pending = {}
        task_iter = iter(tasks)
        exhausted = False

        def _submit_next():
            """Submit next task from iterator. Returns False if exhausted."""
            nonlocal exhausted
            try:
                t = next(task_iter)
                f = executor.submit(process_single_row, t)
                pending[f] = t
                return True
            except StopIteration:
                exhausted = True
                return False

        # Fill initial pool
        for _ in range(min(MAX_INFLIGHT, total_to_do)):
            if not _submit_next():
                break

        while pending:
            # Wait for any future to complete
            done_futures = set()
            for f in as_completed(pending):
                done_futures.add(f)
                break  # process one at a time to refill immediately

            for future in done_futures:
                task_info = pending.pop(future)
                finished_count += 1

                # Refill: submit next task immediately
                if not exhausted:
                    _submit_next()

                try:
                    idx, status, record, v_rows, dim_hits = future.result()
                    log(f"{'─' * 80}")

                    if status == "OK":
                        results.append(record)
                        validation_rows.extend(v_rows)
                        processed_ids.add(str(record['ID']))
                        ok_count += 1
                        log(f"  [{finished_count}/{total_to_do}] OK | {record['ma_ck']} - {record['company_name']} ({record['year']})")
                        log(f"  Words: {record['word_count']:,} | OCR: {record['ocr_pages_used']} | Hits: {record['TOTAL']} (VI={record['TOTAL_vi']}, EN={record['TOTAL_en']})")
                        if dim_hits:
                            for d, info in dim_hits.items():
                                kw_str = ", ".join(info['kws'][:5])
                                if len(info['kws']) > 5:
                                    kw_str += f" (+{len(info['kws'])-5} more)"
                                log(f"     - {d}: {info['total']} hits [{kw_str}]")
                    elif status == "SKIP":
                        skip_count += 1
                        log(f"  [{finished_count}/{total_to_do}] SKIP (no link) | {task_info['ma_ck']}")
                    elif status == "FAIL_PDF":
                        fail_count += 1
                        log(f"  [{finished_count}/{total_to_do}] FAIL_PDF | {task_info['ma_ck']}")
                    else:
                        error_count += 1
                        log(f"  [{finished_count}/{total_to_do}] {status} | {task_info['ma_ck']}")
                    
                    elapsed = time.time() - start_time_all
                    avg = elapsed / finished_count
                    rem = (total_to_do - finished_count) * avg
                    eta_finish = datetime.now() + timedelta(seconds=rem)
                    log(f"  Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
                        f"Avg: {avg:.1f}s/file | "
                        f"ETA: {eta_finish.strftime('%H:%M:%S')} (~{str(timedelta(seconds=int(rem)))} left)")
                    log(f"  Progress: {ok_count} done, {skip_count} skipped, {fail_count} failed, {error_count} errors")

                    # Save partial results every 10 files (reduced I/O overhead vs every 5)
                    if finished_count % 10 == 0:
                        pd.DataFrame(results).to_csv(PARTIAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
                        pd.DataFrame(validation_rows).to_csv(PARTIAL_VALIDATION_PATH, index=False, encoding="utf-8-sig")
                        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                            json.dump({"last_completed_row": idx, "processed_ids": list(processed_ids), "timestamp": str(datetime.now())}, f)

                    # Save log file every LOG_BATCH rows
                    if finished_count % LOG_BATCH == 0:
                        save_log_file(log_batch_start, log_batch_start + LOG_BATCH - 1)
                        log_batch_start += LOG_BATCH

                except Exception as e:
                    error_count += 1
                    log(f"  [CRITICAL ERROR] {e}")

    # Save remaining log buffer
    if log_buffer:
        save_log_file(log_batch_start, log_batch_start + len(log_buffer) - 1)

    # =========================
    # SAVE FINAL OUTPUTS
    # =========================
    results_df = pd.DataFrame(results)
    validation_df = pd.DataFrame(validation_rows)

    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    results_df.to_excel(OUTPUT_XLSX_PATH, index=False)

    validation_df.to_csv(VALIDATION_CSV_PATH, index=False, encoding="utf-8-sig")
    validation_df.to_excel(VALIDATION_XLSX_PATH, index=False)

    total_time = timedelta(seconds=int(time.time() - start_time_all))
    log(f"\n{'=' * 80}")
    log(f"  FINISHED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Total time: {total_time}")
    log(f"  Processed: {ok_count} | Skipped: {skip_count} | Failed: {fail_count} | Errors: {error_count}")
    log(f"  Saved: dt_results.csv, dt_results.xlsx, dt_validation.csv, dt_validation.xlsx")
    log(f"{'=' * 80}")

if __name__ == "__main__":
    run_parallel()
