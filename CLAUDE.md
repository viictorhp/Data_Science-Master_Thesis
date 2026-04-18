# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Music prediction TFM (Trabajo de Fin de Máster). Collects data from Spotify and setlist.fm APIs, performs feature engineering, trains ML models, and exposes results via a Streamlit dashboard. MySQL is available as an optional persistence layer.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows (bash)

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env .env.local  # then fill in real keys
```

`.env` requires: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SETLIST_FM_API_KEY`, and optionally MySQL credentials (`DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`).

## Common Commands

```bash
# Run tests
pytest tests/

# Launch Streamlit dashboard
streamlit run scripts/<dashboard_script>.py

# Run data collection
python -m src.data_collectors.<module>

# Run a specific test
pytest tests/test_<module>.py::test_<name>
```

## Architecture

```
src/
  data_collectors/   # Spotify + setlist.fm API clients, raw data ingestion
  features/          # Feature engineering and transformation pipelines
  models/            # scikit-learn model training, evaluation, serialization
  utils/             # Shared helpers (logging, DB connection, config loading)
config/              # Configuration constants and environment loading
scripts/             # Entry-point scripts (training runs, dashboard, ETL)
tests/               # Unit and integration tests
```

**Data flow**: `data_collectors` → raw data (CSV/JSON, ignored by git) → `features` → processed features → `models` → trained artifacts (`.pkl`/`.joblib`, ignored by git) → `scripts/` dashboard via Streamlit.

**External APIs**:
- Spotify: accessed via `spotipy` library using client credentials flow
- setlist.fm: REST API with `SETLIST_FM_API_KEY` header

**Database**: SQLAlchemy + `mysql-connector-python`; optional — the pipeline can run without MySQL using local files.

## Key Conventions

- Load environment variables with `python-dotenv` (`load_dotenv()`) at the top of any script that needs credentials.
- Trained models and all data files (`data/`, `*.csv`, `*.json`, `*.pkl`) are gitignored — never commit them.
- `src/` is a proper package; use absolute imports (`from src.utils import ...`).
