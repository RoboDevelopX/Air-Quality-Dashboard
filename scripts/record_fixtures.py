"""Record real API responses into tests/fixtures/ for offline tests.

Usage: python scripts/record_fixtures.py [--openaq]
Saves response bodies only (never request headers, so API keys stay out of git).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from aqd.providers.base import WEATHER_COLUMNS
from aqd.providers.open_meteo import (
    AIR_QUALITY_URL,
    AQ_VARIABLES,
    FORECAST_URL,
    GEOCODING_URL,
)
from aqd.providers.openaq import BASE_URL as OPENAQ_URL

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DELHI: dict[str, Any] = {"latitude": 28.6519, "longitude": 77.2315}
COMMON: dict[str, Any] = {
    "timezone": "GMT",
    "timeformat": "unixtime",
    "past_days": 2,
    "forecast_days": 2,
}


def save(name: str, resp: httpx.Response) -> Any:
    body = resp.json()
    (FIXTURES / name).write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n")
    print(f"{name}: HTTP {resp.status_code}, {len(resp.content)} bytes")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openaq", action="store_true", help="also record OpenAQ fixtures")
    args = parser.parse_args()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30) as client:
        aq: dict[str, Any] = {
            **DELHI,
            **COMMON,
            "hourly": ",".join(AQ_VARIABLES),
            "domains": "cams_global",
        }
        save("open_meteo_air_quality.json", client.get(AIR_QUALITY_URL, params=aq))
        weather: dict[str, Any] = {
            **DELHI,
            **COMMON,
            "hourly": ",".join(WEATHER_COLUMNS),
            "wind_speed_unit": "ms",
        }
        save("open_meteo_weather.json", client.get(FORECAST_URL, params=weather))
        geo: dict[str, Any] = {"name": "Delhi", "count": 2, "language": "en"}
        save("open_meteo_geocoding.json", client.get(GEOCODING_URL, params=geo))
        empty: dict[str, Any] = {"name": "Qxzqxzqxz", "count": 2, "language": "en"}
        save("open_meteo_geocoding_empty.json", client.get(GEOCODING_URL, params=empty))
        bad: dict[str, Any] = {**DELHI, "hourly": "pm25"}
        save("open_meteo_error_400.json", client.get(AIR_QUALITY_URL, params=bad))

        if args.openaq:
            key = os.environ.get("OPENAQ_API_KEY")
            if not key:
                print("OPENAQ_API_KEY is not set", file=sys.stderr)
                return 1
            headers = {"X-API-Key": key}
            params: dict[str, Any] = {
                "coordinates": "28.6519,77.2315",
                "radius": 25000,
                "limit": 50,
            }
            locs = save(
                "openaq_locations.json",
                client.get(f"{OPENAQ_URL}/locations", params=params, headers=headers),
            )
            sensor_id = next(
                s["id"]
                for loc in locs["results"]
                for s in loc.get("sensors", [])
                if s["parameter"]["name"] == "pm25"
            )
            save(
                "openaq_sensor_hours.json",
                client.get(
                    f"{OPENAQ_URL}/sensors/{sensor_id}/hours",
                    params={"limit": 48},
                    headers=headers,
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
