"""Feature engineering for direct multi-horizon PM2.5 forecasting.

One row per (forecast origin t, horizon h). The target is PM2.5 at t+h. Features fall in
two groups, and the split is what keeps the model honest:

- Known at the origin (pollutant lags and rolling statistics): computed only from hours
  <= t. `tests/unit/test_features.py` checks this by perturbing everything after t.
- Known in advance for the target hour (calendar, weather *forecast*, horizon): these
  describe t+h but don't use any pollutant observation after t.

The target is learned as a log-ratio to the current value, log1p(y[t+h]) - log1p(y[t]),
so a single global model transfers across cities with very different pollution levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqd.core.averaging import to_hourly_grid

TARGET = "pm2_5"
MAX_HORIZON = 48
LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
ROLL_WINDOWS = (3, 6, 24, 72)
AUX_POLLUTANTS = ("pm10", "no2", "co", "o3")
WEATHER_FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "boundary_layer_height",
)


def origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known at each origin hour t, from pollutant data at or before t only."""
    pm = np.log1p(df[TARGET].clip(lower=0))
    out = pd.DataFrame(index=df.index)
    out["pm_log_0"] = pm
    for lag in LAGS:
        out[f"pm_log_lag{lag}"] = pm.shift(lag)
    for w in ROLL_WINDOWS:
        roll = pm.rolling(w, min_periods=max(2, w // 2))
        out[f"pm_log_mean{w}"] = roll.mean()
        out[f"pm_log_std{w}"] = roll.std()
        out[f"pm_log_max{w}"] = roll.max()
    out["pm_log_diff1"] = pm.diff(1)
    out["pm_log_diff24"] = pm.diff(24)
    for col in AUX_POLLUTANTS:
        if col in df.columns:
            aux = np.log1p(df[col].clip(lower=0))
            out[f"{col}_log_0"] = aux
            out[f"{col}_log_mean24"] = aux.rolling(24, min_periods=12).mean()
        else:
            out[f"{col}_log_0"] = np.nan
            out[f"{col}_log_mean24"] = np.nan
    for col in ("wind_speed_10m", "boundary_layer_height", "relative_humidity_2m"):
        out[f"wx0_{col}"] = df[col] if col in df.columns else np.nan
    return out


def _calendar(times: pd.DatetimeIndex, tz: str) -> dict[str, np.ndarray]:
    local = times.tz_convert(tz)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    dow = local.day_of_week.to_numpy()
    doy = local.day_of_year.to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_weekend": (dow >= 5).astype(float),
    }


def build_frame(
    df: pd.DataFrame,
    tz: str,
    horizons: tuple[int, ...] | range = range(1, MAX_HORIZON + 1),
    origins: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Stack (origin, horizon) rows with features, target and baseline predictions.

    `df` is hourly with pollutant (µg/m³) and optional weather columns on a UTC index.
    Weather at the target hour must be a forecast available at the origin; in training it
    comes from Open-Meteo's historical *forecast* archive for exactly that reason.

    Returned columns: feature columns (see `feature_columns`), plus `origin`, `valid_time`,
    `horizon`, `y` (actual PM2.5 at t+h, may be NaN), `persistence`, `seasonal_naive`.
    """
    grid = to_hourly_grid(df)
    if grid.empty:
        return pd.DataFrame()
    base = origin_features(grid)
    pm = grid[TARGET]
    index = grid.index
    assert isinstance(index, pd.DatetimeIndex)
    origin_idx = index if origins is None else index.intersection(origins)
    pos = index.get_indexer(origin_idx)
    n = len(index)

    frames = []
    for h in horizons:
        target_pos = pos + h
        valid_target = target_pos < n
        tp = np.where(valid_target, target_pos, 0)
        valid_time = origin_idx + pd.Timedelta(hours=h)
        part = base.iloc[pos].reset_index(drop=True)
        part["origin"] = origin_idx
        part["valid_time"] = valid_time
        part["horizon"] = h
        y = pm.to_numpy()[tp]
        part["y"] = np.where(valid_target, y, np.nan)
        # Baselines, both from data at or before the origin.
        part["persistence"] = pm.to_numpy()[pos]
        days_back = int(np.ceil(h / 24))
        seasonal_pos = pos + h - 24 * days_back
        part["seasonal_naive"] = np.where(
            seasonal_pos >= 0, pm.to_numpy()[np.clip(seasonal_pos, 0, None)], np.nan
        )
        part["pm_log_seasonal"] = np.log1p(np.clip(part["seasonal_naive"], 0, None))
        for name, values in _calendar(pd.DatetimeIndex(valid_time), tz).items():
            part[f"tgt_{name}"] = values
        for col in WEATHER_FEATURES:
            if col in grid.columns:
                w = grid[col].to_numpy()[tp]
                part[f"tgt_{col}"] = np.where(valid_target, w, np.nan)
            else:
                part[f"tgt_{col}"] = np.nan
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    wd = np.deg2rad(out.pop("tgt_wind_direction_10m"))
    out["tgt_wind_u"] = np.sin(wd) * out["tgt_wind_speed_10m"]
    out["tgt_wind_v"] = np.cos(wd) * out["tgt_wind_speed_10m"]
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"origin", "valid_time", "y", "persistence", "seasonal_naive", "city"}
    return [c for c in frame.columns if c not in excluded]


def target_delta(frame: pd.DataFrame) -> pd.Series:
    """Training target: log-ratio of future PM2.5 to the value at the origin."""
    delta: pd.Series = np.log1p(frame["y"].clip(lower=0)) - frame["pm_log_0"]
    return delta


def invert_delta(frame: pd.DataFrame, delta: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.expm1(frame["pm_log_0"].to_numpy() + delta).clip(min=0)
    return out
