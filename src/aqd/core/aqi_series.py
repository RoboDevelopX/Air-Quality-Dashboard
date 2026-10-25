"""AQI over time: hourly ("current-style") and daily series from hourly concentrations.

Input frames have a UTC DatetimeIndex and µg/m³ columns named after `Pollutant` values
(pm2_5, pm10, no2, o3, so2, co). Missing columns are treated as unavailable.

Hourly AQI follows how each authority reports "current" air quality:
- US EPA (AirNow): NowCast for PM2.5/PM10, 8-hour O3 and CO, 1-hour NO2 and SO2.
- India NAQI (CPCB live dashboard): trailing 24-hour averages; 8-hour for O3 and CO.

Daily AQI follows each standard's daily definition (24-hour means, daily maximum
8-hour or 1-hour values) grouped by local calendar day. A day needs the same coverage
as a 24-hour average (18 hours for EPA, 16 for NAQI), so partial days at the edges of
the data are left out rather than reported from a few hours.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aqd.core.aqi import Concentrations, InsufficientDataError, compute_aqi
from aqd.core.averaging import (
    MIN_HOURS_8H,
    MIN_HOURS_24H_EPA,
    MIN_HOURS_24H_NAQI,
    nowcast,
    rolling_mean,
    to_hourly_grid,
)
from aqd.core.categories import AqiStandard
from aqd.core.pollutants import Pollutant

_FIELDS = tuple(Concentrations.__dataclass_fields__)


def _col(df: pd.DataFrame, pollutant: Pollutant) -> pd.Series:
    if pollutant.value in df.columns:
        return df[pollutant.value].astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def averaged_inputs(df: pd.DataFrame, standard: AqiStandard) -> pd.DataFrame:
    """Per-hour averaged concentrations (µg/m³) as each standard's 'current' AQI needs."""
    grid = to_hourly_grid(df)
    pm25, pm10 = _col(grid, Pollutant.PM2_5), _col(grid, Pollutant.PM10)
    o3, co = _col(grid, Pollutant.O3), _col(grid, Pollutant.CO)
    no2, so2 = _col(grid, Pollutant.NO2), _col(grid, Pollutant.SO2)
    out = pd.DataFrame(index=grid.index, columns=list(_FIELDS), dtype=float)
    out["o3_8h"] = rolling_mean(o3, 8, MIN_HOURS_8H)
    out["co_8h"] = rolling_mean(co, 8, MIN_HOURS_8H)
    if standard == "us_epa":
        out["pm2_5_24h"] = nowcast(pm25)
        out["pm10_24h"] = nowcast(pm10)
        out["o3_1h"] = o3
        out["no2_1h"] = no2
        out["so2_1h"] = so2
        out["so2_24h"] = rolling_mean(so2, 24, MIN_HOURS_24H_EPA)
    else:
        n = MIN_HOURS_24H_NAQI
        out["pm2_5_24h"] = rolling_mean(pm25, 24, n)
        out["pm10_24h"] = rolling_mean(pm10, 24, n)
        out["no2_24h"] = rolling_mean(no2, 24, n)
        out["so2_24h"] = rolling_mean(so2, 24, n)
    return out


def daily_inputs(df: pd.DataFrame, standard: AqiStandard, tz: str) -> pd.DataFrame:
    """Per local day concentrations following each standard's daily definition."""
    grid = to_hourly_grid(df)
    min_24 = MIN_HOURS_24H_EPA if standard == "us_epa" else MIN_HOURS_24H_NAQI
    assert isinstance(grid.index, pd.DatetimeIndex)
    local_day = grid.index.tz_convert(tz).date

    def daily_mean(s: pd.Series) -> pd.Series:
        g = s.groupby(local_day)
        return g.mean().where(g.count() >= min_24)

    def daily_max(s: pd.Series) -> pd.Series:
        g = s.groupby(local_day)
        return g.max().where(g.count() >= min_24)

    pm25, pm10 = _col(grid, Pollutant.PM2_5), _col(grid, Pollutant.PM10)
    o3, co = _col(grid, Pollutant.O3), _col(grid, Pollutant.CO)
    no2, so2 = _col(grid, Pollutant.NO2), _col(grid, Pollutant.SO2)
    out = pd.DataFrame(
        {
            "pm2_5_24h": daily_mean(pm25),
            "pm10_24h": daily_mean(pm10),
            "o3_8h": daily_max(rolling_mean(o3, 8, MIN_HOURS_8H)),
            "co_8h": daily_max(rolling_mean(co, 8, MIN_HOURS_8H)),
            "so2_24h": daily_mean(so2),
        }
    )
    if standard == "us_epa":
        out["o3_1h"] = daily_max(o3)
        out["no2_1h"] = daily_max(no2)
        out["so2_1h"] = daily_max(so2)
    else:
        out["no2_24h"] = daily_mean(no2)
    out.index = pd.Index(out.index, name="date")
    return out.reindex(columns=list(_FIELDS))


def aqi_from_inputs(inputs: pd.DataFrame, standard: AqiStandard) -> pd.DataFrame:
    """Overall AQI, category and dominant pollutant for each row of averaged inputs.

    Rows that don't meet the standard's data rules get NaN AQI and empty labels.
    """
    rows: list[dict[str, object]] = []
    for record in inputs.itertuples(index=False):
        values = {f: _clean(getattr(record, f)) for f in _FIELDS}
        row: dict[str, object] = {"aqi": np.nan, "category": "", "dominant": ""}
        try:
            result = compute_aqi(Concentrations(**values), standard)
        except InsufficientDataError:
            rows.append(row)
            continue
        row.update(
            aqi=float(result.value),
            category=result.category.key,
            dominant=result.dominant.value,
            **{f"sub_{p.value}": float(si.value) for p, si in result.sub_indices.items()},
        )
        rows.append(row)
    out = pd.DataFrame(rows, index=inputs.index)
    for p in Pollutant:
        if f"sub_{p.value}" not in out.columns:
            out[f"sub_{p.value}"] = np.nan
    return out


def hourly_aqi(df: pd.DataFrame, standard: AqiStandard) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["aqi", "category", "dominant"])
    return aqi_from_inputs(averaged_inputs(df, standard), standard)


def daily_aqi(df: pd.DataFrame, standard: AqiStandard, tz: str = "UTC") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["aqi", "category", "dominant"])
    return aqi_from_inputs(daily_inputs(df, standard, tz), standard)


def _clean(value: object) -> float | None:
    if value is None:
        return None
    f = float(value)  # type: ignore[arg-type]
    return None if math.isnan(f) or f < 0 else f


def latest_valid(frame: pd.DataFrame, column: str = "aqi") -> pd.Series | None:
    """The most recent row with a non-NaN value in `column`, or None."""
    valid = frame[frame[column].notna()]
    return None if valid.empty else valid.iloc[-1]
