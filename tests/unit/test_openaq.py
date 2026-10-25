from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from aqd.providers.base import Location, Provider
from aqd.providers.http import HttpClient, ProviderError
from aqd.providers.openaq import BASE_URL, OpenAQProvider, parse_hours
from aqd.storage.cache import SqliteCache
from tests.conftest import FakeClock

DELHI = Location("Delhi", 28.65195, 77.23149, "Asia/Kolkata")


@pytest.fixture
def http(tmp_path: Path, clock: FakeClock) -> HttpClient:
    return HttpClient(SqliteCache(tmp_path / "c.sqlite3", clock=clock), clock=clock)


def test_requires_api_key(http: HttpClient) -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAQProvider(http, "")


@respx.mock
def test_nearest_sensors_picks_closest_station_per_pollutant(
    http: HttpClient, fixture: Callable[[str], Any]
) -> None:
    route = respx.get(f"{BASE_URL}/locations").mock(
        return_value=httpx.Response(200, json=fixture("openaq_locations.json"))
    )
    provider = OpenAQProvider(http, "secret", radius_m=99_999)
    sensors = provider.nearest_sensors(DELHI)
    request = route.calls[0].request
    assert request.headers["X-API-Key"] == "secret"
    assert request.url.params["coordinates"] == "28.6519,77.2315"  # latitude,longitude
    assert request.url.params["radius"] == "25000"  # clamped to the documented max
    assert {p.value: s for p, s in sensors.items()} == {
        "pm10": (13843, "ITO, Delhi - CPCB"),
        "pm2_5": (23534, "R K Puram, Delhi - DPCC"),
    }  # ozone in ppm is skipped


@respx.mock
def test_air_quality_returns_history_only(http: HttpClient, fixture: Callable[[str], Any]) -> None:
    respx.get(f"{BASE_URL}/locations").mock(
        return_value=httpx.Response(200, json=fixture("openaq_locations.json"))
    )
    respx.get(url__regex=rf"{BASE_URL}/sensors/\d+/hours").mock(
        return_value=httpx.Response(200, json=fixture("openaq_sensor_hours.json"))
    )
    provider = OpenAQProvider(http, "secret")
    assert isinstance(provider, Provider)
    data = provider.air_quality(DELHI, past_days=2)
    assert data.forecast.empty
    assert set(data.history.columns) == {"pm2_5", "pm10"}
    assert len(data.history) == 24
    assert data.source == "openaq"


def test_parse_hours_skips_invalid_rows(fixture: Callable[[str], Any]) -> None:
    series = parse_hours(fixture("openaq_sensor_hours.json"))
    assert len(series) == 24  # negative and null values dropped
    assert str(series.index.tz) == "UTC"
    assert series.index.is_monotonic_increasing


@respx.mock
def test_non_mapping_response_raises(http: HttpClient) -> None:
    respx.get(f"{BASE_URL}/locations").mock(return_value=httpx.Response(200, json=[1, 2]))
    with pytest.raises(ProviderError):
        OpenAQProvider(http, "k").nearest_sensors(DELHI)
