"""In-memory providers for service, pipeline and UI tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from aqd.providers.base import POLLUTANT_COLUMNS, AirQualityData, Location
from aqd.providers.http import ProviderError
from tests.synthetic import city_frame

WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "boundary_layer_height",
]


class FakeAirQuality:
    """Serves synthetic data; 'now' is a fixed hour inside the synthetic series."""

    name = "fake"
    attribution = "Synthetic test data"

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        now: str = "2023-08-01T12:00Z",
        *,
        stale: bool = False,
        fail_for: set[str] | None = None,
    ) -> None:
        self.frame = frame if frame is not None else city_frame(days=300, seed=3, level=60)
        self.now = pd.Timestamp(now)
        self.stale = stale
        self.fail_for = fail_for or set()
        self.calls: list[tuple[str, int, int]] = []

    def air_quality(
        self, location: Location, *, past_days: int, forecast_days: int
    ) -> AirQualityData:
        self.calls.append((location.name, past_days, forecast_days))
        if location.name in self.fail_for:
            raise ProviderError(f"{location.name} unavailable")
        start = self.now.floor("D") - pd.Timedelta(days=past_days)
        end = self.now.floor("D") + pd.Timedelta(days=forecast_days)
        cols = [c for c in POLLUTANT_COLUMNS if c in self.frame.columns]
        window = self.frame.loc[(self.frame.index >= start) & (self.frame.index < end), cols]
        return AirQualityData(
            location=location,
            history=window[window.index <= self.now],
            forecast=window[window.index > self.now],
            source=self.name,
            issued_at=self.now.to_pydatetime(),
            stale=self.stale,
        )

    def air_quality_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        cols = [c for c in POLLUTANT_COLUMNS if c in self.frame.columns]
        return self.frame[cols]


class FakeWeather:
    def __init__(self, frame: pd.DataFrame | None = None, *, fail: bool = False) -> None:
        self.frame = frame if frame is not None else city_frame(days=300, seed=3, level=60)
        self.fail = fail

    def weather(self, location: Location, *, past_days: int, forecast_days: int) -> pd.DataFrame:
        if self.fail:
            raise ProviderError("weather down")
        return self.frame[WEATHER_COLS]

    def weather_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        return self.frame[WEATHER_COLS]


class FakeGeocoder:
    def __init__(self, places: list[Location] | None = None) -> None:
        self.places = places or [
            Location("Delhi", 28.65, 77.23, "Asia/Kolkata", "India", "IN", "Delhi"),
            Location("London", 51.51, -0.13, "Europe/London", "United Kingdom", "GB", "England"),
        ]

    def search(self, name: str, count: int = 5) -> list[Location]:
        q = name.strip().lower()
        return [p for p in self.places if p.name.lower().startswith(q)][:count] if q else []


class FakeStation:
    name = "station"
    attribution = "Synthetic station"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def air_quality(
        self, location: Location, *, past_days: int, forecast_days: int
    ) -> AirQualityData:
        if self.fail:
            raise ProviderError("station down")
        idx = pd.date_range("2023-08-01T10:00Z", periods=3, freq="h")
        hist = pd.DataFrame({"pm2_5": [50.0, 55.0, None]}, index=idx)
        return AirQualityData(location, hist, pd.DataFrame(), "station", datetime.now(UTC))
