from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from aqd.providers.base import POLLUTANT_COLUMNS, Location, Provider
from aqd.providers.http import HttpClient, ProviderError
from aqd.providers.open_meteo import (
    AIR_QUALITY_URL,
    FORECAST_URL,
    GEOCODING_URL,
    HISTORICAL_FORECAST_URL,
    OpenMeteoAirQuality,
    OpenMeteoGeocoder,
    OpenMeteoWeather,
    _chunks,
    parse_hourly,
)
from aqd.storage.cache import SqliteCache
from tests.conftest import FakeClock

DELHI = Location("Delhi", 28.65195, 77.23149, "Asia/Kolkata", "India", "IN", "Delhi")


@pytest.fixture
def http(tmp_path: Path, clock: FakeClock) -> HttpClient:
    return HttpClient(SqliteCache(tmp_path / "c.sqlite3", clock=clock), clock=clock)


@respx.mock
def test_air_quality_parses_and_splits_history_forecast(
    http: HttpClient, clock: FakeClock, fixture: Callable[[str], Any]
) -> None:
    payload = fixture("open_meteo_air_quality.json")
    payload["hourly"]["pm2_5"][5] = None  # the recorded day has no gaps; real ones do
    # Recorded 2026-10-25 with past_days=2: the first hour is 2026-10-23T00:00Z.
    clock.now = payload["hourly"]["time"][0] + 60 * 3600  # 2026-10-25T12:00Z
    route = respx.get(AIR_QUALITY_URL).mock(return_value=httpx.Response(200, json=payload))
    provider = OpenMeteoAirQuality(http)
    assert isinstance(provider, Provider)
    data = provider.air_quality(DELHI, past_days=2, forecast_days=2)

    params = route.calls[0].request.url.params
    assert params["timeformat"] == "unixtime"
    assert params["domains"] == "cams_global"
    assert params["past_days"] == "2"
    assert set(params["hourly"].split(",")) == {
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "carbon_monoxide",
    }

    assert list(data.history.columns) == list(POLLUTANT_COLUMNS)
    assert str(data.history.index.tz) == "UTC"
    # History ends at the clock's hour, forecast starts an hour later.
    assert data.history.index.max() == pd.Timestamp("2026-10-25T12:00Z")
    assert data.forecast.index.min() == pd.Timestamp("2026-10-25T13:00Z")
    assert len(data.history) + len(data.forecast) == 96
    assert pd.isna(data.history["pm2_5"].iloc[5])  # null in the payload stays missing
    assert data.source == "open_meteo"
    assert not data.stale


@respx.mock
def test_air_quality_400_raises_with_reason(
    http: HttpClient, fixture: Callable[[str], Any]
) -> None:
    respx.get(AIR_QUALITY_URL).mock(
        return_value=httpx.Response(400, json=fixture("open_meteo_error_400.json"))
    )
    with pytest.raises(ProviderError, match="Cannot initialize"):
        OpenMeteoAirQuality(http).air_quality(DELHI)


@pytest.mark.parametrize(("past", "fc"), [(-1, 1), (93, 1), (1, 8)])
def test_air_quality_validates_ranges(http: HttpClient, past: int, fc: int) -> None:
    with pytest.raises(ValueError):
        OpenMeteoAirQuality(http).air_quality(DELHI, past_days=past, forecast_days=fc)


@respx.mock
def test_air_quality_range_chunks_by_year(http: HttpClient, fixture: Callable[[str], Any]) -> None:
    route = respx.get(AIR_QUALITY_URL).mock(
        return_value=httpx.Response(200, json=fixture("open_meteo_air_quality.json"))
    )
    frame = OpenMeteoAirQuality(http).air_quality_range(DELHI, date(2023, 1, 1), date(2024, 6, 30))
    assert route.call_count == 2
    first = route.calls[0].request.url.params
    assert (first["start_date"], first["end_date"]) == ("2023-01-01", "2023-12-31")
    assert not frame.index.duplicated().any()  # identical chunks are de-duplicated


def test_parse_hourly_rejects_unexpected_units(fixture: Callable[[str], Any]) -> None:
    payload = fixture("open_meteo_air_quality.json")
    payload["hourly_units"]["pm2_5"] = "ppb"
    with pytest.raises(ProviderError, match="unexpected unit"):
        parse_hourly(payload, {"pm2_5": "pm2_5"}, expect_units={"pm2_5": frozenset({"μg/m³"})})


def test_parse_hourly_missing_variable_becomes_nan(fixture: Callable[[str], Any]) -> None:
    payload = fixture("open_meteo_air_quality.json")
    frame = parse_hourly(payload, {"pm2_5": "pm2_5", "dust": "dust"})
    assert frame["dust"].isna().all()


@pytest.mark.parametrize(
    "payload",
    [{}, {"hourly": {"pm2_5": [1.0]}}, {"hourly": {"time": [0, 3600], "pm2_5": [1.0]}}],
)
def test_parse_hourly_malformed(payload: dict[str, Any]) -> None:
    with pytest.raises(ProviderError):
        parse_hourly(payload, {"pm2_5": "pm2_5"})


@respx.mock
def test_weather_forecast_and_history(http: HttpClient, fixture: Callable[[str], Any]) -> None:
    body = fixture("open_meteo_weather.json")
    fc = respx.get(FORECAST_URL).mock(return_value=httpx.Response(200, json=body))
    hist = respx.get(HISTORICAL_FORECAST_URL).mock(return_value=httpx.Response(200, json=body))
    weather = OpenMeteoWeather(http)
    frame = weather.weather(DELHI, past_days=2, forecast_days=2)
    assert fc.calls[0].request.url.params["wind_speed_unit"] == "ms"
    assert frame["boundary_layer_height"].notna().all()
    assert len(frame) == 96
    weather.weather_range(DELHI, date(2024, 1, 1), date(2024, 1, 4))
    assert hist.call_count == 1


@respx.mock
def test_geocoder_parses_results(http: HttpClient, fixture: Callable[[str], Any]) -> None:
    respx.get(GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=fixture("open_meteo_geocoding.json"))
    )
    results = OpenMeteoGeocoder(http).search(" Delhi ", count=2)
    assert [r.name for r in results] == ["Delhi", "Delhi"]
    delhi = results[0]
    assert delhi.timezone == "Asia/Kolkata"
    assert delhi.country_code == "IN"
    assert delhi.population == 11034555
    assert delhi.label == "Delhi, National Capital Territory of Delhi, India"
    assert delhi.key == "28.652,77.231"
    assert results[1].label == "Delhi, New York, United States"
    assert results[1].timezone == "America/New_York"


def test_location_label_skips_admin1_equal_to_name() -> None:
    assert Location("Delhi", 0, 0, admin1="Delhi", country="India").label == "Delhi, India"


@respx.mock
def test_geocoder_no_results(http: HttpClient, fixture: Callable[[str], Any]) -> None:
    respx.get(GEOCODING_URL).mock(
        return_value=httpx.Response(200, json=fixture("open_meteo_geocoding_empty.json"))
    )
    assert OpenMeteoGeocoder(http).search("Qxzqxz") == []


def test_geocoder_short_query_skips_network(http: HttpClient) -> None:
    assert OpenMeteoGeocoder(http).search(" a ") == []


def test_chunks() -> None:
    assert _chunks(date(2024, 1, 1), date(2024, 1, 1), days=365) == [
        (date(2024, 1, 1), date(2024, 1, 1))
    ]
    with pytest.raises(ValueError):
        _chunks(date(2024, 1, 2), date(2024, 1, 1), days=365)
