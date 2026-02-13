# Install (Local Python)

This app runs locally with Python. No Windows installer is required.

## Requirements
- Python 3.10+
- Playwright browsers

## Setup
```powershell
pip install -r requirements.txt
python -m playwright install
```

## Run
```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Troubleshooting
- If you see a Playwright browser error when scraping or exporting PNGs, run:
  `python -m playwright install`
