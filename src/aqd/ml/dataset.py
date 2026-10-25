"""Training data assembly: one hourly frame per city, stacked into model-ready rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from aqd.ml.features import MAX_HORIZON, build_frame
from aqd.providers.base import HistoryProvider, Location, WeatherProvider

# Training panel: cities spanning very different pollution regimes and climates, so a
# single model learns dynamics that transfer, not one city's levels.
TRAINING_CITIES: tuple[Location, ...] = (
    Location("Delhi", 28.6519, 77.2315, "Asia/Kolkata", "India", "IN"),
    Location("Mumbai", 19.0728, 72.8826, "Asia/Kolkata", "India", "IN"),
    Location("Lahore", 31.5580, 74.3507, "Asia/Karachi", "Pakistan", "PK"),
    Location("Beijing", 39.9075, 116.3972, "Asia/Shanghai", "China", "CN"),
    Location("Cairo", 30.0626, 31.2497, "Africa/Cairo", "Egypt", "EG"),
    Location("Lagos", 6.4541, 3.3947, "Africa/Lagos", "Nigeria", "NG"),
    Location("London", 51.5085, -0.1257, "Europe/London", "United Kingdom", "GB"),
    Location("Los Angeles", 34.0522, -118.2437, "America/Los_Angeles", "United States", "US"),
    Location("São Paulo", -23.5475, -46.6361, "America/Sao_Paulo", "Brazil", "BR"),
    Location("Jakarta", -6.2146, 106.8451, "Asia/Jakarta", "Indonesia", "ID"),
)

# CAMS global data in Open-Meteo starts in August 2022.
DATA_START = date(2022, 8, 1)


@dataclass(frozen=True, slots=True)
class CityData:
    location: Location
    hourly: pd.DataFrame  # pollutants + weather on a UTC index


def fetch_city(
    aq: HistoryProvider,
    weather: WeatherProvider,
    location: Location,
    start: date,
    end: date,
    *,
    observed_until: pd.Timestamp | None = None,
) -> CityData:
    """Hourly pollutants joined with archived weather forecasts for one location.

    A range that reaches today also returns CAMS forecast hours. Pass `observed_until`
    (usually the current hour) to blank pollutant values after it, so forecasts are never
    mistaken for actuals. Weather is kept: it plays the role of a forecast anyway.
    """
    pollution = aq.air_quality_range(location, start, end)
    if observed_until is not None:
        pollution = pollution.copy()
        pollution.loc[pollution.index > observed_until] = np.nan
    wx = weather.weather_range(location, start, end)
    hourly = pollution.join(wx, how="left")
    return CityData(location, hourly)


def stack(
    cities: Mapping[str, CityData],
    *,
    origin_stride_h: int,
    horizons_per_origin: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Stacked (origin, horizon) rows for all cities with a `city` column.

    Origins are taken every `origin_stride_h` hours (aligned to midnight UTC). For training,
    `horizons_per_origin` randomly keeps that many horizons per origin, which cuts memory by
    ~4x at 12 of 48 with no systematic bias. For evaluation pass None to keep all horizons.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for name, city in cities.items():
        idx = pd.DatetimeIndex(city.hourly.index)
        origins = idx[(idx.hour % origin_stride_h) == 0]
        frame = build_frame(city.hourly, city.location.timezone, origins=origins)
        if frame.empty:
            continue
        if horizons_per_origin is not None and horizons_per_origin < MAX_HORIZON:
            keep = rng.random(len(frame)) < horizons_per_origin / MAX_HORIZON
            frame = frame[keep]
        frame.insert(0, "city", name)
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    float_cols = out.select_dtypes("float64").columns
    out[float_cols] = out[float_cols].astype("float32")
    return out
