# Valorant OPGG Stats Generator

Internal tool that scrapes OP.GG VALORANT profiles to build team dashboards and comparison views.

## Features
- Team dashboard with map performance, agent pools, and scouting insights
- Competitive stats with map filtering
- Map pool filters (defaults to current competitive pool)
- Team-to-team comparison dashboards
- PNG snapshot export for team and comparison dashboards

## Quick Start (Local)
```powershell/cmd
pip install -r requirements.txt
python -m playwright install
python app.py
```
Open `http://127.0.0.1:5000` in a browser.

## Usage
1. Upload a team list (Riot IDs).
2. Choose seasons and start a scrape.
3. Use the dashboard filters:
   - Map filters (default = current comp pool)
4. Save or download snapshots for comparisons.

## Map Pool Defaults
Current comp pool is auto-enabled:
- Abyss
- Bind
- Breeze
- Corrode
- Pearl
- Split
- Haven

Use the `Map Filters` button to include additional maps.

## Install (Local)
See `docs/INSTALL.md` for setup and run steps.

## Notes
- Scraping depends on OP.GG layout stability.
- Playwright is required for live scraping.
