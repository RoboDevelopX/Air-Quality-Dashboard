"""Optional OpenAQ v3 provider: ground-station PM measurements near a location.

Enabled only when OPENAQ_API_KEY is set. Station data has no forecast, so `forecast` is
always empty; the app uses it to show the nearest monitor next to the CAMS model values.

Docs (checked 2026-10-25): https://docs.openaq.org/ - auth header `X-API-Key`; free tier
60 requests/minute and 2,000/hour; `coordinates=lat,lon` with `radius` up to 25,000 m.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from aqd.core.pollutants import Pollutant
from aqd.providers.base import AirQualityData, Location
from aqd.providers.http import HttpClient, ProviderError

BASE_URL = "https://api.openaq.org/v3"
MAX_RADIUS_M = 25_000

# OpenAQ parameter names -> our columns. Only units we can use without conversion.
_PARAMETERS: dict[str, Pollutant] = {"pm25": Pollutant.PM2_5, "pm10": Pollutant.PM10}
_UGM3 = frozenset({"µg/m³", "μg/m³", "ug/m3"})


class OpenAQProvider:
    name = "openaq"
    attribution = "Ground station data: OpenAQ (openaq.org) and its data providers."

    def __init__(
        self,
        http: HttpClient,
        api_key: str,
        *,
        radius_m: int = MAX_RADIUS_M,
        ttl_s: float = 3600,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAQ requires an API key (OPENAQ_API_KEY)")
        self._http = http
        self._headers = {"X-API-Key": api_key}
        self._radius_m = min(radius_m, MAX_RADIUS_M)
        self._ttl_s = ttl_s

    def nearest_sensors(self, location: Location) -> dict[Pollutant, tuple[int, str]]:
        """For each supported pollutant, the (sensor id, station name) closest to `location`."""
        params = {
            "coordinates": f"{location.latitude:.4f},{location.longitude:.4f}",
            "radius": self._radius_m,
            "limit": 50,
        }
        payload = self._get(f"{BASE_URL}/locations", params)
        stations = sorted(payload.get("results") or [], key=lambda r: r.get("distance") or 1e12)
        found: dict[Pollutant, tuple[int, str]] = {}
        for station in stations:
            for sensor in station.get("sensors") or []:
                param = sensor.get("parameter") or {}
                pollutant = _PARAMETERS.get(str(param.get("name", "")).lower())
                if pollutant and pollutant not in found and param.get("units") in _UGM3:
                    found[pollutant] = (int(sensor["id"]), str(station.get("name") or ""))
        return found

    def air_quality(
        self, location: Location, *, past_days: int = 7, forecast_days: int = 0
    ) -> AirQualityData:
        sensors = self.nearest_sensors(location)
        now = datetime.now(UTC)
        start = now - timedelta(days=past_days)
        columns: dict[str, pd.Series] = {}
        for pollutant, (sensor_id, _) in sensors.items():
            params = {
                "datetime_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "datetime_to": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": 1000,
            }
            payload = self._get(f"{BASE_URL}/sensors/{sensor_id}/hours", params)
            columns[pollutant.value] = parse_hours(payload)
        history = pd.DataFrame(columns) if columns else pd.DataFrame()
        if not history.empty:
            history.index.name = "time"
        return AirQualityData(
            location=location,
            history=history,
            forecast=pd.DataFrame(),
            source=self.name,
            issued_at=now,
        )

    def _get(self, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        data = self._http.get_json(url, params, ttl_s=self._ttl_s, headers=self._headers).data
        if not isinstance(data, Mapping):
            raise ProviderError("unexpected OpenAQ response")
        return data


def parse_hours(payload: Mapping[str, Any]) -> pd.Series:
    """Hourly sensor values indexed by the UTC start of each hour."""
    times: list[pd.Timestamp] = []
    values: list[float] = []
    for row in payload.get("results") or []:
        try:
            ts = pd.Timestamp(row["period"]["datetimeFrom"]["utc"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if value < 0:  # instruments occasionally report negative noise
            continue
        times.append(ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC"))
        values.append(value)
    series = pd.Series(values, index=pd.DatetimeIndex(times, name="time"), dtype=float)
    return series[~series.index.duplicated(keep="last")].sort_index()
