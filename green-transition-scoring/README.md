# Green Transition LLM Scoring Pipeline

This project scores green transition disclosure in Vietnamese annual reports and sustainability reports.

## What it does

- Reads company report links from a Google Sheet CSV export
- Prioritizes sustainability reports over annual reports when available
- Downloads PDF reports
- Extracts text using PDF parsing and OCR
- Splits reports into text chunks
- Retrieves relevant chunks for each environmental indicator
- Uses an LLM to score indicators based on a fixed rubric
- Optionally runs a judge step to validate selected scores
- Saves summary scores, indicator-level details, and error logs

## Main files

- `main.py`: main Python script
- `requirements.txt`: required Python packages
- `.gitignore`: ignored files

## Required environment variables

```bash
export DATASET_URL="your_google_sheet_csv_export_link"
export LLM_API_KEY="your_api_key"