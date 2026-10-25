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
mypy                                                # type check (strict, src/aqd + scripts)
pre-commit run --all-files                          # everything CI runs (activate the venv first:
                                                    # the mypy hook uses the venv's mypy)

python scripts/train.py                             # fetch data, CV-evaluate, write models/ + reports/
python scripts/train.py --cities Delhi London --folds 3 --no-sarimax   # quick run
python scripts/snapshot_forecasts.py                # save the provider's current forecast to snapshots/
python scripts/compare_provider.py                  # score saved model vs CAMS snapshots (needs actuals)
python scripts/record_fixtures.py                   # re-record test fixtures from the live APIs

docker build -t air-quality-dashboard . && docker run --rm -p 8501:8501 air-quality-dashboard
```

## Layout

```
app.py              Streamlit entry point; only calls aqd.ui.app.main()
src/aqd/
  config.py         Settings dataclass from AQD_* env vars (+ optional .env)
  log.py            JSON structured logging
  providers/        base.py (Provider/Geocoder/WeatherProvider protocols + Location),
                    http.py (retries, cache, stale fallback), open_meteo.py (air quality,
                    weather, geocoding), openaq.py (optional)
  storage/          cache.py: SQLite TTL cache
  core/             pollutants, aqi (breakpoint tables), averaging (NowCast), aqi_series,
                    categories, health, anomaly, service (DashboardService -> view models)
  ml/               features, cv, metrics, models (LightGBM + SARIMAX), evaluate, dataset,
                    train (pipeline + provider comparison), inference
  ui/               app.py (layout/state), components.py, charts.py (Plotly builders)
scripts/            train.py, snapshot_forecasts.py, compare_provider.py, record_fixtures.py
models/             Committed model artifact (LightGBM text + JSON); app runs without it
reports/            metrics.json, results.md, plots written by scripts/train.py
snapshots/          Provider forecast snapshots (written by the scheduled workflow)
tests/              unit/, integration/ (pipeline e2e, Streamlit AppTest), fixtures/,
                    fakes.py (in-memory providers), synthetic.py (synthetic series)
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
- **The app must start without a usable model.** LightGBM is imported lazily in
  `ml/models.py`, and `ml/inference.load_forecaster` returns None when the artifact is missing
  or the native library won't load (macOS without `brew install libomp`). Don't add a
  module-level `import lightgbm` anywhere the app imports; a test checks this.

## Conventions

- Type hints everywhere; `mypy --strict` must pass. Use `from __future__ import annotations`.
- Timestamps are timezone-aware UTC internally (`pd.Timestamp(..., tz="UTC")`); convert to
  local time only for display.
- Concentrations are µg/m³ everywhere except inside `core/aqi.py`, which converts gases to
  ppb/ppm as each standard requires.
- Tests never hit the network. Use recorded fixtures in `tests/fixtures/` with respx, or the
  fakes in `tests/fakes.py`. Record new fixtures with `scripts/record_fixtures.py`.
- UI tests use `streamlit.testing.v1.AppTest` and monkeypatch `aqd.ui.app.build_service`.
- Don't add scikit-learn back without a use: metrics are a few lines, and CV needs a
  target-time filter that `TimeSeriesSplit` can't express on stacked multi-horizon rows.
- Secrets only via env vars (`OPENAQ_API_KEY`). Never log them; `Settings` hides them in repr.
- One logical change per commit, imperative subject line ("Add NAQI breakpoint table").

## External APIs

Before changing provider code, re-read the official docs; do not rely on memory.
- Open-Meteo Air Quality: https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo Geocoding: https://open-meteo.com/en/docs/geocoding-api
- Open-Meteo Historical Forecast (weather): https://open-meteo.com/en/docs/historical-forecast-api
- OpenAQ v3 (optional): https://docs.openaq.org/ (header X-API-Key; 60/min, 2,000/h;
  `coordinates=lat,lon`, radius <= 25,000 m)

Open-Meteo free tier: non-commercial, < 10,000 calls/day, 5,000/h, 600/min; CC BY 4.0.
Air-quality history is archived CAMS forecasts (not stations) and past forecasts are not
archived, hence `scripts/snapshot_forecasts.py`.

Checked against live responses on 2026-10-25 (the docs pages' source is also on GitHub at
`open-meteo/open-meteo-website`, under `src/routes/en/docs/`):
- Concentration units come back as `μg/m³` with the Greek mu (U+03BC); CO is µg/m³ too.
- `domains=cams_global` is accepted; CAMS global history starts in August 2022.
- Geocoding with no match returns `{"generationtime_ms": ...}` with no `results` key.
- `boundary_layer_height` is valid on both weather APIs, but the historical-forecast archive
  only has values from 2024-09-01 (null before). LightGBM treats them as missing.
- An invalid variable gives HTTP 400 `{"error": true, "reason": "..."}`.
- Under load the APIs sometimes drop TLS handshakes, return 429, or cut a body short;
  `HttpClient` retries all three.
- The archive is stitched from each run's first hours, so a saved CAMS forecast equals the
  "actual" until the next run. Keep that in mind when comparing against the provider.
