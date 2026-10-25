# Model card: PM2.5 48-hour forecaster

Numbers come from `reports/metrics.json` (training run of 25 Oct 2026, LightGBM 4.7.0).

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
  height. The archive stitches together the first hours of each model run, so the weather
  for a target hour t+h is a forecast made shortly before t+h, not one made at the origin t.
  Training and evaluation therefore see better weather than the app does at long horizons
  (see the weather ablation below). Boundary layer height is only archived from
  2024-09-01; it is missing (NaN, which LightGBM handles natively) before that, and present
  in every CV test fold and at inference.
- **Cities**: Delhi, Mumbai, Lahore, Beijing, Cairo, Lagos, London, Los Angeles, São Paulo,
  Jakarta. They span very clean to very polluted air and tropical to temperate climates.
- **Window**: 1 Aug 2022 to 23 Oct 2026, hourly (about 37,000 hours per city). The final
  model trains on 1.1 million sampled (origin, horizon) rows.

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

Held-out period: 26 Apr to 23 Oct 2026, 343,840 forecasts across 10 cities. MAE in µg/m³.

| Model | MAE 1-6h | MAE 7-24h | MAE 25-48h | RMSE (all) | Skill vs persistence (all) |
|---|---|---|---|---|---|
| Persistence | 10.20 | 17.73 | 18.85 | 29.40 | 0% |
| Seasonal naive | 13.14 | 13.15 | 15.20 | 23.89 | 18% |
| LightGBM | 6.61 | 10.15 | 11.21 | 17.78 | 41% |

On the 86,140 rows where SARIMAX also ran (one origin per city per day):

| Model | MAE 1-6h | MAE 7-24h | MAE 25-48h | RMSE (all) | Skill vs persistence (all) |
|---|---|---|---|---|---|
| Persistence | 12.67 | 17.14 | 19.08 | 28.46 | 0% |
| Seasonal naive | 12.20 | 13.46 | 15.20 | 23.88 | 19% |
| SARIMAX | 7.28 | 12.30 | 13.02 | 20.57 | 31% |
| LightGBM | 5.86 | 10.50 | 11.17 | 17.98 | 42% |

- **By horizon**: persistence is better at 1 h ahead (MAE 3.66 vs 4.20); LightGBM wins from
  2 h onwards (`by_horizon` in `metrics.json`).
- **By city** (skill vs persistence): Jakarta 57%, Beijing 47%, São Paulo 43%, Lagos 42%,
  London 36%, Lahore 33%, Los Angeles 33%, Cairo 30%, Delhi 26%, Mumbai 21%.
- **Interval coverage**: the 80% intervals contained 81.8% of held-out values. Intervals are
  per-horizon residual quantiles from out-of-fold predictions (split conformal). They're the
  same width in clean and dirty conditions apart from the log scale, so expect under-coverage
  during episodes.

**Weather ablation.** Retraining and re-evaluating with every weather input removed gives 33%
skill overall (1-6 h 32%, 7-24 h 36%, 25-48 h 32%) instead of 41% (35%, 43%, 40%). Because
the archived training weather is sharper than a live 1-2 day forecast, live skill at long
horizons probably lies between those two numbers. The Delhi peak of 17 Oct 2026 shows the
effect: with weather the model anticipates it 24 h ahead
(`reports/predictions_delhi_24h.png`); without weather it peaks a day late, like persistence.

Plots: `reports/error_by_horizon.png`, `reports/predictions_*_24h.png`.

**Against the CAMS forecast**: not yet measured. `scripts/compare_provider.py` writes
`reports/provider_comparison.md` from snapshots issued after the training window, once their
actuals exist; the first snapshot is from 25 Oct 2026. The comparison favours CAMS: the
"actuals" are stitched from CAMS's own runs, so within one run (about 12 h) the provider's
forecast equals the actual. Only the 25-48 h bucket is a reasonably fair comparison.

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
- **Seasonal coverage of the evaluation.** The held-out months are late April to late October.
  The North Indian, Pakistani and Chinese winter smog season (Nov-Feb) is in training but not
  in the test window, so the reported errors understate winter errors there.
- **Optimistic weather inputs.** See the ablation above.

## Ethical and safety notes

Air quality affects health decisions. The app labels forecasts with their evaluated skill
and an interval, says when the model is no better than "no change", and points users to local
authorities. It doesn't present the forecast as more certain than the evaluation supports.
