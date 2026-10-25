# Evaluation results

Rolling-origin CV, 6 folds x 30 days, 48h gap; held-out period 2026-04-26 to 2026-10-23. Errors are PM2.5 in µg/m³ against CAMS (not station measurements).

80% interval coverage on held-out folds: 82%.

## All held-out rows (baselines vs LightGBM)

| group | model | n | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|---|
| 1-6h | persistence | 43190 | 10.20 | 19.25 | 0.000 |
| 1-6h | seasonal naive | 43190 | 13.14 | 22.37 | -0.289 |
| 1-6h | LightGBM | 43190 | 6.61 | 11.93 | 0.352 |
| 25-48h | persistence | 171440 | 18.85 | 31.14 | 0.000 |
| 25-48h | seasonal naive | 171440 | 15.20 | 25.32 | 0.193 |
| 25-48h | LightGBM | 171440 | 11.21 | 19.18 | 0.405 |
| 7-24h | persistence | 129210 | 17.73 | 29.83 | 0.000 |
| 7-24h | seasonal naive | 129210 | 13.15 | 22.39 | 0.258 |
| 7-24h | LightGBM | 129210 | 10.15 | 17.49 | 0.428 |
| all | persistence | 343840 | 17.34 | 29.40 | 0.000 |
| all | seasonal naive | 343840 | 14.17 | 23.89 | 0.183 |
| all | LightGBM | 343840 | 10.24 | 17.78 | 0.410 |

## Daily origins where SARIMAX was also run (all four models, same rows)

| group | model | n | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|---|
| 1-6h | persistence | 10800 | 12.67 | 23.97 | 0.000 |
| 1-6h | seasonal naive | 10800 | 12.20 | 20.17 | 0.038 |
| 1-6h | SARIMAX | 10800 | 7.28 | 12.87 | 0.426 |
| 1-6h | LightGBM | 10800 | 5.86 | 10.63 | 0.538 |
| 25-48h | persistence | 42950 | 19.08 | 30.14 | 0.000 |
| 25-48h | seasonal naive | 42950 | 15.20 | 25.31 | 0.204 |
| 25-48h | SARIMAX | 42950 | 13.02 | 21.84 | 0.318 |
| 25-48h | LightGBM | 42950 | 11.17 | 19.25 | 0.415 |
| 7-24h | persistence | 32390 | 17.14 | 27.51 | 0.000 |
| 7-24h | seasonal naive | 32390 | 13.46 | 23.05 | 0.215 |
| 7-24h | SARIMAX | 32390 | 12.30 | 20.90 | 0.282 |
| 7-24h | LightGBM | 32390 | 10.50 | 18.18 | 0.388 |
| all | persistence | 86140 | 17.55 | 28.46 | 0.000 |
| all | seasonal naive | 86140 | 14.17 | 23.88 | 0.193 |
| all | SARIMAX | 86140 | 12.03 | 20.57 | 0.314 |
| all | LightGBM | 86140 | 10.25 | 17.98 | 0.416 |

## By city (all horizons)

| group | model | n | mae | rmse | skill_vs_persistence |
|---|---|---|---|---|---|
| Beijing | persistence | 34384 | 35.62 | 47.84 | 0.000 |
| Beijing | seasonal naive | 34384 | 28.85 | 39.19 | 0.190 |
| Beijing | LightGBM | 34384 | 18.89 | 25.82 | 0.470 |
| Cairo | persistence | 34384 | 6.29 | 8.74 | 0.000 |
| Cairo | seasonal naive | 34384 | 6.16 | 8.55 | 0.021 |
| Cairo | LightGBM | 34384 | 4.38 | 6.21 | 0.304 |
| Delhi | persistence | 34384 | 29.63 | 42.22 | 0.000 |
| Delhi | seasonal naive | 34384 | 28.94 | 41.07 | 0.023 |
| Delhi | LightGBM | 34384 | 21.96 | 32.70 | 0.259 |
| Jakarta | persistence | 34384 | 41.57 | 52.79 | 0.000 |
| Jakarta | seasonal naive | 34384 | 21.92 | 29.83 | 0.473 |
| Jakarta | LightGBM | 34384 | 18.08 | 24.77 | 0.565 |
| Lagos | persistence | 34384 | 7.64 | 12.17 | 0.000 |
| Lagos | seasonal naive | 34384 | 6.44 | 10.72 | 0.156 |
| Lagos | LightGBM | 34384 | 4.46 | 7.56 | 0.416 |
| Lahore | persistence | 34384 | 22.30 | 30.67 | 0.000 |
| Lahore | seasonal naive | 34384 | 20.36 | 28.89 | 0.087 |
| Lahore | LightGBM | 34384 | 14.91 | 21.51 | 0.331 |
| London | persistence | 34384 | 4.99 | 7.57 | 0.000 |
| London | seasonal naive | 34384 | 5.23 | 7.72 | -0.048 |
| London | LightGBM | 34384 | 3.19 | 4.71 | 0.361 |
| Los Angeles | persistence | 34384 | 6.06 | 8.05 | 0.000 |
| Los Angeles | seasonal naive | 34384 | 4.77 | 6.47 | 0.213 |
| Los Angeles | LightGBM | 34384 | 4.08 | 5.47 | 0.327 |
| Mumbai | persistence | 34384 | 6.14 | 10.67 | 0.000 |
| Mumbai | seasonal naive | 34384 | 5.79 | 9.73 | 0.057 |
| Mumbai | LightGBM | 34384 | 4.87 | 8.39 | 0.206 |
| São Paulo | persistence | 34384 | 13.18 | 19.59 | 0.000 |
| São Paulo | seasonal naive | 34384 | 13.26 | 19.41 | -0.006 |
| São Paulo | LightGBM | 34384 | 7.53 | 11.45 | 0.428 |
| all | persistence | 343840 | 17.34 | 29.40 | 0.000 |
| all | seasonal naive | 343840 | 14.17 | 23.89 | 0.183 |
| all | LightGBM | 343840 | 10.24 | 17.78 | 0.410 |
