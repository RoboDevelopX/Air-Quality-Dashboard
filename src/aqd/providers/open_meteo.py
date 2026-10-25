"""Open-Meteo providers: air quality (CAMS), weather, and geocoding.

Docs (checked 2026-10-25):
- Air quality: https://open-meteo.com/en/docs/air-quality-api
- Forecast / historical forecast: https://open-meteo.com/en/docs,
  https://open-meteo.com/en/docs/historical-forecast-api
- Geocoding: https://open-meteo.com/en/docs/geocoding-api

Free tier: non-commercial use, < 10,000 calls/day, 5,000/hour, 600/minute; data under
CC BY 4.0. Air quality data comes from CAMS (Copernicus Atmosphere Monitoring Service).
Past air-quality hours are archived forecasts stitched from each run's first hours, not
station measurements.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from aqd.core.pollutants import Pollutant
from aqd.providers.base import WEATHER_COLUMNS, AirQualityData, Location
from aqd.providers.http import HttpClient, ProviderError

log = logging.getLogger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

ATTRIBUTION = (
    "Air quality: Copernicus Atmosphere Monitoring Service (CAMS) via Open-Meteo.com, "
    "CC BY 4.0. Weather and geocoding: Open-Meteo.com (GeoNames), CC BY 4.0."
)

# Open-Meteo variable name -> our column name.
AQ_VARIABLES: dict[str, Pollutant] = {
    "pm2_5": Pollutant.PM2_5,
    "pm10": Pollutant.PM10,
    "nitrogen_dioxide": Pollutant.NO2,
    "ozone": Pollutant.O3,
    "sulphur_dioxide": Pollutant.SO2,
    "carbon_monoxide": Pollutant.CO,
}
# Both the Greek mu (U+03BC) and the micro sign (U+00B5) appear in the wild.
_UGM3_UNITS = frozenset({"μg/m³", "µg/m³", "ug/m3"})

MAX_PAST_DAYS = 92
MAX_AQ_FORECAST_DAYS = 7


def parse_hourly(
    payload: Mapping[str, Any],
    variables: Mapping[str, str],
    *,
    expect_units: Mapping[str, frozenset[str]] | None = None,
) -> pd.DataFrame:
    """Turn an Open-Meteo `hourly` block (timeformat=unixtime) into a UTC-indexed frame.

    `variables` maps API names to output column names; variables missing from the payload
    become NaN columns. Unit checks guard against silent upstream unit changes.
    """
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or "time" not in hourly:
        raise ProviderError("Open-Meteo response has no hourly data")
    units = payload.get("hourly_units") or {}
    for api_name, allowed in (expect_units or {}).items():
        unit = units.get(api_name)
        if api_name in hourly and unit is not None and unit not in allowed:
            raise ProviderError(f"unexpected unit for {api_name}: {unit!r}")
    index = pd.to_datetime(pd.Series(hourly["time"], dtype="int64"), unit="s", utc=True)
    frame = pd.DataFrame(index=pd.DatetimeIndex(index, name="time"))
    for api_name, column in variables.items():
        values = hourly.get(api_name)
        if values is None:
            frame[column] = float("nan")
        else:
            if len(values) != len(frame):
                raise ProviderError(f"length mismatch for {api_name}")
            frame[column] = pd.to_numeric(pd.Series(values, index=frame.index), errors="coerce")
    return frame.sort_index()


class OpenMeteoAirQuality:
    """Provider + HistoryProvider backed by the Open-Meteo Air Quality API (CAMS)."""

    name = "open_meteo"
    attribution = ATTRIBUTION

    def __init__(
        self,
        http: HttpClient,
        *,
        domain: str = "cams_global",
        ttl_forecast_s: float = 3600,
        ttl_history_s: float = 6 * 3600,
    ) -> None:
        self._http = http
        self._domain = domain
        self._ttl_forecast_s = ttl_forecast_s
        self._ttl_history_s = ttl_history_s

    def _params(self, location: Location) -> dict[str, Any]:
        return {
            "latitude": round(location.latitude, 4),
            "longitude": round(location.longitude, 4),
            "hourly": ",".join(AQ_VARIABLES),
            "domains": self._domain,
            "timezone": "GMT",
            "timeformat": "unixtime",
        }

    def _parse(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        variables = {k: v.value for k, v in AQ_VARIABLES.items()}
        return parse_hourly(payload, variables, expect_units={k: _UGM3_UNITS for k in AQ_VARIABLES})

    def air_quality(
        self, location: Location, *, past_days: int = 30, forecast_days: int = 3
    ) -> AirQualityData:
        if not 0 <= past_days <= MAX_PAST_DAYS:
            raise ValueError(f"past_days must be 0..{MAX_PAST_DAYS}")
        if not 0 <= forecast_days <= MAX_AQ_FORECAST_DAYS:
            raise ValueError(f"forecast_days must be 0..{MAX_AQ_FORECAST_DAYS}")
        params = self._params(location) | {"past_days": past_days, "forecast_days": forecast_days}
        result = self._http.get_json(AIR_QUALITY_URL, params, ttl_s=self._ttl_forecast_s)
        frame = self._parse(result.data)
        issued = datetime.fromtimestamp(result.fetched_at, tz=UTC)
        # Split at the hour the data was fetched, so a cached response stays consistent.
        cutoff = pd.Timestamp(issued).floor("h")
        return AirQualityData(
            location=location,
            history=frame[frame.index <= cutoff],
            forecast=frame[frame.index > cutoff],
            source=self.name,
            issued_at=issued,
            stale=result.stale,
        )

    def air_quality_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        """Hourly history between two dates (inclusive), fetched in yearly chunks."""
        frames = []
        for chunk_start, chunk_end in _chunks(start, end, days=365):
            params = self._params(location) | {
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
            }
            result = self._http.get_json(AIR_QUALITY_URL, params, ttl_s=self._ttl_history_s)
            frames.append(self._parse(result.data))
        frame = pd.concat(frames)
        return frame[~frame.index.duplicated(keep="last")]


class OpenMeteoWeather:
    """WeatherProvider: forecast API for the app, historical-forecast API for training."""

    def __init__(
        self,
        http: HttpClient,
        *,
        variables: tuple[str, ...] = WEATHER_COLUMNS,
        ttl_forecast_s: float = 3600,
        ttl_history_s: float = 6 * 3600,
    ) -> None:
        self._http = http
        self._variables = variables
        self._ttl_forecast_s = ttl_forecast_s
        self._ttl_history_s = ttl_history_s

    def _params(self, location: Location) -> dict[str, Any]:
        return {
            "latitude": round(location.latitude, 4),
            "longitude": round(location.longitude, 4),
            "hourly": ",".join(self._variables),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "timeformat": "unixtime",
        }

    def _parse(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        return parse_hourly(payload, {v: v for v in self._variables})

    def weather(
        self, location: Location, *, past_days: int = 30, forecast_days: int = 3
    ) -> pd.DataFrame:
        params = self._params(location) | {"past_days": past_days, "forecast_days": forecast_days}
        result = self._http.get_json(FORECAST_URL, params, ttl_s=self._ttl_forecast_s)
        return self._parse(result.data)

    def weather_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        frames = []
        for chunk_start, chunk_end in _chunks(start, end, days=365):
            params = self._params(location) | {
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
            }
            result = self._http.get_json(HISTORICAL_FORECAST_URL, params, ttl_s=self._ttl_history_s)
            frames.append(self._parse(result.data))
        frame = pd.concat(frames)
        return frame[~frame.index.duplicated(keep="last")]


class OpenMeteoGeocoder:
    def __init__(self, http: HttpClient, *, ttl_s: float = 30 * 86400) -> None:
        self._http = http
        self._ttl_s = ttl_s

    def search(self, name: str, count: int = 5) -> list[Location]:
        """Matching places, best first. Fewer than 2 characters never matches (per docs)."""
        query = name.strip()
        if len(query) < 2:
            return []
        params = {"name": query, "count": max(1, min(count, 100)), "language": "en"}
        payload = self._http.get_json(GEOCODING_URL, params, ttl_s=self._ttl_s).data
        return [_location(r) for r in (payload.get("results") or []) if _valid_result(r)]


def _valid_result(r: Mapping[str, Any]) -> bool:
    return all(k in r for k in ("name", "latitude", "longitude"))


def _location(r: Mapping[str, Any]) -> Location:
    return Location(
        name=str(r["name"]),
        latitude=float(r["latitude"]),
        longitude=float(r["longitude"]),
        timezone=str(r.get("timezone") or "UTC"),
        country=r.get("country") or None,
        country_code=r.get("country_code") or None,
        admin1=r.get("admin1") or None,
        population=int(r["population"]) if r.get("population") else None,
        id=int(r["id"]) if r.get("id") is not None else None,
    )


def _chunks(start: date, end: date, *, days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end must not be before start")
    out = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out
