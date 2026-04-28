# Digital Transformation Keyword Extraction

This project extracts digital transformation keyword mentions from Vietnamese company annual reports.

## What it does

- Reads company report links from a Google Sheet
- Downloads annual report PDFs
- Extracts text from PDFs
- Uses OCR when scanned pages do not have enough text
- Counts Vietnamese and English keywords by digital transformation dimension
- Saves results into CSV and Excel files
- Saves validation snippets for checking keyword evidence

## Main files

- `main.py`: main Python script
- `keyword_dictionary-ficc.csv`: keyword dictionary
- `requirements.txt`: required Python packages
- `.gitignore`: files ignored by GitHub

## Output files

The script creates:

- `dt_results.csv`
- `dt_results.xlsx`
- `dt_validation.csv`
- `dt_validation.xlsx`

These output files are ignored by GitHub by default.

## Note

Tesseract OCR must be installed separately before running the script.