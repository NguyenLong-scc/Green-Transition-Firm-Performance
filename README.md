# Green Transition and Firm Performance in Vietnam

Research pipeline that turns Vietnamese annual and sustainability reports (PDFs) into firm-year measures of **digital transformation** and **green transition** disclosure, which are then used to study firm performance.

The repo contains two independent pipelines. Both read the same list of company reports (ID, year, report links) and each produces a firm-year dataset keyed by `ID` and `year`.

| Folder | What it measures | Method | Output |
|---|---|---|---|
| [`digital-transformation-keyword-extraction/`](digital-transformation-keyword-extraction) | Digital-transformation keyword mentions in annual reports | PDF text extraction, OCR fallback, dictionary matching (Aho-Corasick) | `dt_results`, `dt_validation` |
| [`green-transition-scoring/`](green-transition-scoring) | Green-transition disclosure quality (8 primary / 24 secondary indicators, score range -1 to 46) | PDF text extraction, OCR fallback, chunking, TF-IDF retrieval, LLM scoring, selective LLM judge | scores, indicator details, errors |

## Overview

```mermaid
flowchart TD
    S["Google Sheet (CSV export)<br/>ID, year, company, report links"]
    S --> P1
    S --> P2

    subgraph P1["Pipeline 1: Digital-transformation keyword extraction"]
        direction TB
        A1["Annual report PDFs"] --> A2["Text extraction<br/>(native, OCR fallback)"]
        A2 --> A3["Dictionary matching<br/>(4 dimensions, Vietnamese and English)"]
        A3 --> A4["Firm-year keyword counts<br/>and rates per 1,000 words"]
    end

    subgraph P2["Pipeline 2: Green-transition scoring"]
        direction TB
        B1["Sustainability report (preferred)<br/>or annual report PDFs"] --> B2["Text extraction<br/>(native, OCR fallback)"]
        B2 --> B3["Chunk and retrieve<br/>evidence per indicator"]
        B3 --> B4["LLM rubric scoring<br/>and selective judge"]
        B4 --> B5["Firm-year green scores<br/>and indicator details"]
    end

    A4 --> M["Merge on ID and year<br/>into firm-level panel dataset"]
    B5 --> M
    M --> R["Regression analysis<br/>(Python and Stata)<br/>technology adoption vs firm performance"]
```

> The merge and regression steps happen downstream of these scripts and are not part of this repo.

## Pipeline 1: Digital-transformation keyword extraction

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        I1["DATASET_URL<br/>CSV with ID, company_name,<br/>ma_ck, year, annual_report link"]
        I2["keyword_dictionary-ficc.csv<br/>Dimension, Keyword_en, Keyword_vi"]
    end

    I2 --> K1["Load and normalize keywords<br/>lowercase, strip punctuation,<br/>deduplicate"]
    K1 --> K2["Build Aho-Corasick automaton<br/>4 dimensions: AI Technology,<br/>Block Chain Technology,<br/>Cloud Computing Tech,<br/>Digital Tech Application"]

    I1 --> Q1["Build task list<br/>skip IDs already processed<br/>(progress.json + partial CSVs)"]
    Q1 --> W1

    subgraph W1["Parallel workers (ProcessPoolExecutor, MAX_WORKERS=4)"]
        direction TB
        D1["Convert Google Drive link<br/>to direct download URL"] --> D2["Download PDF<br/>3 retries, check PDF header"]
        D2 -->|"no link"| S1["Status: SKIP"]
        D2 -->|"download failed"| S2["Status: FAIL_PDF"]
        D2 -->|"valid PDF"| E1["Extract text page by page<br/>(PyMuPDF)"]
        E1 --> E2{"Page has at least<br/>300 characters?"}
        E2 -->|"yes"| E3["Use native text"]
        E2 -->|"no"| E4["OCR at 300 DPI<br/>(Tesseract, vie+eng)<br/>autocontrast and sharpen"]
        E3 --> C1["Clean text<br/>lowercase, Vietnamese-aware regex,<br/>collapse whitespace, word count"]
        E4 --> C1
        C1 --> M1["Single-pass keyword scan<br/>(Aho-Corasick)"]
        M1 --> M2["Word-boundary filter"]
        M2 --> M3["Binary count per keyword<br/>(0 or 1, not frequency)"]
        M3 --> M4["Aggregate by dimension<br/>VI count, EN count, total,<br/>matched keywords, rate per 1,000 words"]
        M3 --> M5["Save validation snippet<br/>(first match, 80 chars of context)"]
    end

    K2 --> W1
    M4 --> O1
    M5 --> O2
    W1 -->|"every 10 files"| CP["Checkpoint<br/>partial CSVs + progress.json"]

    O1["dt_results.csv / .xlsx<br/>one row per company-year"]
    O2["dt_validation.csv / .xlsx<br/>keyword evidence snippets"]
```

## Pipeline 2: Green-transition scoring

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        I1["DATASET_URL<br/>CSV with ID, year,<br/>annual_report, sustainability_report"]
        I2["LLM_API_KEY<br/>OpenAI-compatible API"]
    end

    I1 --> L1["Pick report<br/>sustainability report first,<br/>else annual report"]
    L1 --> L2["Download PDF<br/>(3 retries)"]
    L2 --> X1{"Cached text<br/>in cache_text/?"}
    X1 -->|"yes"| X3["Load cached pages"]
    X1 -->|"no"| X2["Extract pages (PyMuPDF)<br/>OCR if a page has fewer than 40 words<br/>(Tesseract, 240 DPI, vie+eng)"]
    X2 --> X4["Save page text to cache"]
    X3 --> G1
    X4 --> G1
    G1{"At least 80 words<br/>of text?"}
    G1 -->|"no"| ER["Log error<br/>(errors CSV)"]
    G1 -->|"yes"| C1["Split into chunks<br/>150 words, 30 overlap,<br/>track page range"]

    subgraph RET["Retrieval (local, no LLM)"]
        direction TB
        T1["TF-IDF index over chunks<br/>(1-2 word n-grams)"] --> T2["Per indicator: build queries<br/>English + Vietnamese names,<br/>keyword hints, scoring-type terms"]
        T2 --> T3["Cosine similarity, best score<br/>across queries"]
        T3 --> T4["Boost chunks with numbers and units<br/>(quantitative indicators)<br/>or violation terms (penalty indicator)"]
        T4 --> T5["Keep top 2 chunks<br/>per indicator"]
    end
    C1 --> T1

    subgraph SC["LLM scoring"]
        direction TB
        R1["Rubric: 8 primary groups,<br/>24 secondary indicators"] --> R2["Batch by scoring type<br/>4 LLM calls per report<br/>(max 10 evidence chunks per call)"]
        T5 --> R2
        I2 --> R2
        R2 --> R3["Scorer returns JSON<br/>score, confidence, reason,<br/>evidence chunk IDs, quote"]
        R3 --> R4["Snap invalid scores<br/>to the allowed scale"]
    end

    R4 --> J1{"Flag for judge?<br/>low confidence, flagged by scorer,<br/>or score 2 / -1"}
    J1 -->|"yes (max 1 per report)"| J2["LLM judge<br/>KEEP, LOWER, or RAISE<br/>with grounding confidence"]
    J1 -->|"no"| V1
    J2 --> V1

    V1["Validation checks per indicator<br/>allowed score, evidence chunks,<br/>evidence quote, retrieval score at least 0.02"]
    V1 --> V2["Status: OK, REVIEW, or FAIL"]
    V2 --> A1["Aggregate<br/>subtotal per primary indicator,<br/>total score (range -1 to 46)"]

    A1 --> O1["green_transition_scores_budget_vi<br/>.csv / .xlsx"]
    V2 --> O2["green_transition_indicator_details_budget_vi<br/>.csv / .xlsx"]
    ER --> O3["green_transition_errors_budget_vi.csv"]
    A1 -->|"every 50 reports"| CP["Partial saves<br/>(merged and deduplicated)"]
```

### Scoring rubric

| Scoring type | Allowed scores | Primary indicators (secondary count) |
|---|---|---|
| Disclosure (binary) | 0 or 2 | Environmental Management (4), Information Disclosure Carriers (3) |
| Yes / No (binary) | 0 or 2 | Environmental Certification (4), Environmental Responsibility (1) |
| Description scale | 0 (none), 1 (qualitative), 2 (quantitative) | Environmental Liabilities (5), Performance and Governance (4), Green Innovation (2) |
| Negative penalty | -1 or 0 | Environmental Activities: violations, accidents, petitions (1) |

## Setup

Both pipelines need [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed with the Vietnamese language data.

```bash
# Pipeline 1
cd digital-transformation-keyword-extraction
pip install -r requirements.txt
export DATASET_URL="your_google_sheet_csv_export_link"
python main.py

# Pipeline 2
cd green-transition-scoring
pip install -r requirements.txt
export DATASET_URL="your_google_sheet_csv_export_link"
export LLM_API_KEY="your_api_key"
python main.py
```

Optional environment variables: `TESSERACT_CMD`, `TESSDATA_DIR`, `MAX_WORKERS` (pipeline 1), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_MODEL_SCORER`, `LLM_MODEL_JUDGE` (pipeline 2).
