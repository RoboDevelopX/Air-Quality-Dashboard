"""Provider interfaces and the domain types they return.

Providers hide each API's quirks: callers get UTC-indexed pandas frames with columns named
after `Pollutant` values in µg/m³, never raw JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from aqd.core.pollutants import Pollutant

POLLUTANT_COLUMNS: tuple[str, ...] = tuple(p.value for p in Pollutant)

WEATHER_COLUMNS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "boundary_layer_height",
)


@dataclass(frozen=True, slots=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str = "UTC"
    country: str | None = None
    country_code: str | None = None
    admin1: str | None = None
    population: int | None = None
    id: int | None = None

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)

    @property
    def key(self) -> str:
        """Stable identifier for caching and comparison lists."""
        return f"{self.latitude:.3f},{self.longitude:.3f}"


@dataclass(frozen=True, slots=True)
class AirQualityData:
    """Hourly concentrations around now: `history` up to the current hour, `forecast` after.

    Frames have a UTC DatetimeIndex and a subset of POLLUTANT_COLUMNS (µg/m³).
    """

    location: Location
    history: pd.DataFrame
    forecast: pd.DataFrame
    source: str
    issued_at: datetime  # when the provider data was fetched (UTC)
    stale: bool = False  # served from an expired cache entry after a failed fetch


@runtime_checkable
class Provider(Protocol):
    """An air quality data source. Open-Meteo is required; others are optional."""

    name: str
    attribution: str

    def air_quality(
        self, location: Location, *, past_days: int, forecast_days: int
    ) -> AirQualityData: ...


class HistoryProvider(Protocol):
    """Long hourly history for model training."""

    def air_quality_range(self, location: Location, start: date, end: date) -> pd.DataFrame: ...


class Geocoder(Protocol):
    def search(self, name: str, count: int = 5) -> list[Location]: ...


class WeatherProvider(Protocol):
    def weather(self, location: Location, *, past_days: int, forecast_days: int) -> pd.DataFrame:
        """Hourly weather (WEATHER_COLUMNS subset), UTC index, past and forecast together."""
        ...

    def weather_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        """Archived *forecast* weather for training (what was knowable at the time)."""
        ...
