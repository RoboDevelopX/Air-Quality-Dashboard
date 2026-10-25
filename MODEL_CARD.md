# Model card: PM2.5 48-hour forecaster

<!-- Numbers marked TBD are filled from reports/metrics.json after scripts/train.py runs. -->

## Overview

| | |
|---|---|
| Task | Forecast hourly PM2.5 (µg/m³) 1 to 48 hours ahead for any location |
| Model | Global LightGBM regressor, direct multi-horizon (horizon is a feature) |
| Target | log1p(PM2.5 at t+h) − log1p(PM2.5 at t), the log-ratio to the current value |
| Loss | L1 (median), matching the MAE headline metric and robust to spikes |
| Artifact | `models/pm25_lgbm.txt` (LightGBM text) + `models/pm25_lgbm.json` (features, intervals, metrics) |
| Intended use | A portfolio demo of a forecasting pipeline and a rough 2-day outlook. **Not** for health, regulatory or safety decisions |

## Data

- **Source**: Open-Meteo Air Quality API, CAMS global domain (about 0.4°/45 km, 3-hourly
  interpolated to hourly), from August 2022. Past hours are CAMS's own short-range forecasts
  stitched together, not station measurements.
- **Weather**: Open-Meteo Historical Forecast API (archived short-range forecasts): temperature,
  humidity, precipitation, wind speed/direction, surface pressure, cloud cover, boundary layer
  height. Using archived forecasts instead of observations keeps training inputs similar to
  what's available when forecasting live.
- **Cities**: Delhi, Mumbai, Lahore, Beijing, Cairo, Lagos, London, Los Angeles, São Paulo,
  Jakarta. They span very clean to very polluted air and tropical to temperate climates.
- **Window**: TBD (data_start to data_end in `reports/metrics.json`).

## Features (all computed at forecast origin t unless marked target-time)

- PM2.5 (log) at t and lags 1, 2, 3, 6, 12, 24, 48, 168 h; rolling mean, std and max over
  3, 6, 24, 72 h; 1 h and 24 h differences.
- PM10, NO₂, CO, O₃ (log) at t and their 24 h means.
- Wind speed, boundary layer height and humidity at t.
- Seasonal reference: PM2.5 at the same clock hour on the most recent day at or before t.
- Target-time: local hour, weekday and day-of-year (sin/cos), weekend flag; weather forecast
  for t+h (wind as u/v components); the horizon h.

**Leakage controls**: a unit test replaces every pollutant value after t with random noise
and checks that every origin feature stays identical. CV keeps a 48 h gap between the last
training origin and the test window, and also drops any training row whose target time falls
in the test window.

## Evaluation

Rolling-origin CV with an expanding window: 6 folds × 30 days covering the most recent
6 months, 48 h gap. Test origins every 6 h with all 48 horizons; SARIMAX runs at one origin
per day per city (each origin needs its own fit), and all four models are also compared on
that same subset.

| Model | MAE 1-6h | MAE 7-24h | MAE 25-48h | RMSE (all) | Skill vs persistence (all) |
|---|---|---|---|---|---|
| Persistence | TBD | TBD | TBD | TBD | 0% |
| Seasonal naive | TBD | TBD | TBD | TBD | TBD |
| SARIMAX (daily subset) | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |

80% interval coverage on held-out folds: TBD (target 80%). Intervals are per-horizon
residual quantiles from earlier folds (split conformal). They're the same width in clean and
dirty conditions apart from the log scale, so expect under-coverage during episodes.

Plots: `reports/error_by_horizon.png`, `reports/predictions_*_24h.png`.

**Against the CAMS forecast**: see `reports/provider_comparison.md`. Only snapshots issued after
the training window are used. CAMS is scored against its own later analysis, which favours it.

## Limitations and failure modes

- **The ground truth is a model.** Error against real monitors will be larger, and CAMS has
  known regional biases (for example, it often underestimates winter PM2.5 in the
  Indo-Gangetic Plain).
- **Episodes.** Wildfire smoke, dust storms, stubble burning and festival fireworks are
  under-represented. The model reacts late and under-predicts peaks.
- **Unseen regimes.** Cities unlike the training panel (high altitude, polar, small islands)
  may behave differently. The global model has no city identity feature, by design.
- **Weather forecast quality.** At longer horizons the model depends on the weather
  forecast. If it is unavailable in the app, the model runs without it and is less accurate.
- **Distribution shift.** CAMS model upgrades change the data's behaviour, so retrain
  periodically and watch live skill using the snapshots.
- **PM2.5 only.** The forecast AQI ignores other pollutants.

## Ethical and safety notes

Air quality affects health decisions. The app labels forecasts with their evaluated skill
and an interval, says when the model is no better than "no change", and points users to local
authorities. It doesn't present the forecast as more certain than the evaluation supports.
