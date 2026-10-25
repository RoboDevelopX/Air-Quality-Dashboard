# CLAUDE.md

Guidance for anyone (human or AI) working in this repo. Keep it current when commands,
structure or conventions change.

## What this is

A Streamlit dashboard showing current, historical and forecast air quality for any city,
with a PM2.5 forecasting model at its core. Portfolio project: correctness, honesty about
model limits, and clean engineering matter more than feature count.

## Stack

Python 3.11+, Streamlit, Plotly, pandas, httpx, scikit-learn, LightGBM, statsmodels
(training only), SQLite (stdlib) for caching, pytest + respx, ruff, mypy (strict), pre-commit,
Docker, GitHub Actions.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                             # app + training + dev tools
pre-commit install                                  # once per clone

streamlit run app.py                                # run the app (http://localhost:8501)
pytest                                              # all tests (no network access needed)
pytest --cov                                        # tests + coverage (fails under 80%)
ruff check . && ruff format --check .               # lint + format check
mypy                                                # type check (strict, src/aqd only)
pre-commit run --all-files                          # everything CI runs, locally
```

## Layout

```
app.py              Streamlit entry point; only calls aqd.ui.app.main()
src/aqd/
  config.py         Settings dataclass from AQD_* env vars (+ optional .env)
  log.py            JSON structured logging
  providers/        Data sources behind the Provider protocol (Open-Meteo, optional OpenAQ)
  storage/          SQLite TTL cache
  core/             Pure domain logic: AQI math, categories, health advice, anomalies, service
  ml/               Features, baselines, models, time-series CV, evaluation, inference
  ui/               Streamlit rendering only
scripts/            Offline entry points (training, fixture recording)
models/             Committed model artifact + metrics (regenerate with scripts/train.py)
tests/unit|integration|fixtures
```

## Architecture rules

- **No business logic in `ui/`.** UI code receives view-model dataclasses from
  `core/service.py` and renders them. Anything worth testing lives outside `ui/`.
- **Providers return domain types**, never raw JSON. Parsing lives in the provider module.
  New sources implement `providers.base.Provider`; the app must work with only Open-Meteo.
- **`core/` and `ml/` are pure**: no network, no Streamlit, no global state. Pass data in.
- **No look-ahead.** Every feature at forecast origin `t` uses data at or before `t` only;
  `tests/unit/test_features.py` enforces this. CV folds keep a gap of at least the max horizon.
- **The app loads a saved model artifact**; it never trains. Training is `scripts/train.py`.

## Conventions

- Type hints everywhere; `mypy --strict` must pass. Use `from __future__ import annotations`.
- Timestamps are timezone-aware UTC internally (`pd.Timestamp(..., tz="UTC")`); convert to
  local time only for display.
- Concentrations are µg/m³ everywhere except inside `core/aqi.py`, which converts gases to
  ppb/ppm as each standard requires.
- Tests never hit the network. Use recorded fixtures in `tests/fixtures/` with respx.
  Record new ones with `scripts/record_fixtures.py`.
- Secrets only via env vars (`OPENAQ_API_KEY`). Never log them; `Settings` hides them in repr.
- One logical change per commit, imperative subject line ("Add NAQI breakpoint table").

## External APIs

Before changing provider code, re-read the official docs; do not rely on memory.
- Open-Meteo Air Quality: https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo Geocoding: https://open-meteo.com/en/docs/geocoding-api
- Open-Meteo Historical Forecast (weather): https://open-meteo.com/en/docs/historical-forecast-api
- OpenAQ v3 (optional): https://docs.openaq.org/
