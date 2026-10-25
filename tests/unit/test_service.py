from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqd.core.pollutants import Pollutant
from aqd.core.service import MAX_COMPARE, DashboardService
from aqd.ml.dataset import CityData, stack
from aqd.ml.inference import Forecaster
from aqd.ml.models import LgbmForecaster
from aqd.providers.base import Location
from tests.fakes import FakeAirQuality, FakeGeocoder, FakeStation, FakeWeather
from tests.synthetic import city_frame

DELHI = Location("Delhi", 28.65, 77.23, "Asia/Kolkata", "India", "IN", "Delhi")


@pytest.fixture(scope="module")
def forecaster() -> Forecaster:
    frame = city_frame(days=200, seed=3, level=60)
    data = {"X": CityData(DELHI, frame)}
    train = stack(data, origin_stride_h=6, horizons_per_origin=12)
    model = LgbmForecaster.fit(train, num_boost_round=60, params={"num_leaves": 15})
    model.intervals = {h: (-0.3, 0.3) for h in range(1, 49)}
    model.metadata = {
        "summary": {
            "cv": {
                "interval_coverage_80": 0.79,
                "buckets": {"1-6h": {"skill_vs_persistence": 0.2}},
            }
        }
    }
    return Forecaster(model)


def test_dashboard_full(forecaster: Forecaster) -> None:
    svc = DashboardService(
        FakeAirQuality(), FakeGeocoder(), weather=FakeWeather(), forecaster=forecaster
    )
    d = svc.dashboard(DELHI, "us_epa", history_days=7)
    assert d.standard_name == "US EPA AQI"
    assert d.current is not None
    assert d.current.time == pd.Timestamp("2023-08-01T12:00Z")
    assert 0 < d.current.aqi < 300
    assert d.current.advice.summary
    readings = {r.pollutant: r for r in d.current.pollutants}
    assert set(readings) == set(Pollutant)
    assert readings[Pollutant.CO].unit == "mg/m³"
    assert readings[Pollutant.NO2].averaging == "1-hour"
    # History is trimmed to the requested window, daily to the requested days.
    assert d.history.index.min() > pd.Timestamp("2023-07-25T12:00Z")
    assert len(d.daily) <= 7
    fc = d.forecast
    assert fc.model_available and fc.message is None
    assert len(fc.frame) == 48
    assert fc.frame["model"].notna().all() and fc.frame["provider"].notna().all()
    assert (fc.frame["model_lo"] <= fc.frame["model_hi"]).all()
    assert fc.frame["model_aqi"].notna().all()
    assert fc.frame["model_category"].str.len().gt(0).all()
    assert fc.skill == {"1-6h": 0.2}
    assert fc.interval_coverage == 0.79
    assert fc.peak is not None
    assert d.warnings == []
    assert not d.stale


def test_dashboard_without_model_or_weather() -> None:
    svc = DashboardService(FakeAirQuality(), FakeGeocoder())
    d = svc.dashboard(DELHI, "in_naqi")
    assert not d.forecast.model_available
    assert "model isn't available" in (d.forecast.message or "")
    assert d.forecast.frame["provider"].notna().any()
    assert d.forecast.frame["model_aqi"].isna().all()
    assert d.forecast.frame["provider_aqi"].notna().any()
    assert d.current is not None and d.current.category.key in {
        "good",
        "satisfactory",
        "moderate",
        "poor",
        "very_poor",
        "severe",
    }


def test_stale_and_weather_failure_become_warnings(forecaster: Forecaster) -> None:
    svc = DashboardService(
        FakeAirQuality(stale=True),
        FakeGeocoder(),
        weather=FakeWeather(fail=True),
        forecaster=forecaster,
    )
    d = svc.dashboard(DELHI, "us_epa")
    assert d.stale
    assert any("saved copy" in w for w in d.warnings)
    assert any("without weather" in w for w in d.warnings)
    assert d.forecast.model_available  # still forecasts without weather


def test_naqi_insufficient_data_message() -> None:
    frame = city_frame(days=30)[["pm2_5", "no2"]]  # only two pollutants
    svc = DashboardService(FakeAirQuality(frame, now="2023-01-20T12:00Z"), FakeGeocoder())
    d = svc.dashboard(DELHI, "in_naqi", history_days=7)
    assert d.current is None
    assert "three pollutants" in (d.current_message or "")
    # The same data is enough for EPA, which has no minimum pollutant count.
    assert svc.dashboard(DELHI, "us_epa", history_days=7).current is not None


def test_empty_history() -> None:
    frame = city_frame(days=10)
    svc = DashboardService(FakeAirQuality(frame, now="2030-01-01T00:00Z"), FakeGeocoder())
    d = svc.dashboard(DELHI, "us_epa", history_days=7)
    assert d.current is None
    assert "No recent" in (d.current_message or "")
    assert d.forecast.frame.empty
    assert d.forecast.peak is None


def test_short_history_model_message(forecaster: Forecaster) -> None:
    frame = city_frame(days=2)
    svc = DashboardService(
        FakeAirQuality(frame, now="2023-01-02T03:00Z"), FakeGeocoder(), forecaster=forecaster
    )
    d = svc.dashboard(DELHI, "us_epa", history_days=1)
    assert not d.forecast.model_available
    assert "can't forecast" in (d.forecast.message or "")


def test_history_days_validated() -> None:
    svc = DashboardService(FakeAirQuality(), FakeGeocoder())
    with pytest.raises(ValueError):
        svc.dashboard(DELHI, "us_epa", history_days=0)


def test_spikes_surface_in_dashboard() -> None:
    frame = city_frame(days=300, seed=3, level=60)
    spike_at = pd.Timestamp("2023-07-31T20:00Z")
    frame.loc[spike_at : spike_at + pd.Timedelta(hours=2), "pm2_5"] += 400
    svc = DashboardService(FakeAirQuality(frame), FakeGeocoder())
    d = svc.dashboard(DELHI, "us_epa", history_days=7)
    assert d.spikes
    assert d.spikes[-1].start == spike_at
    assert d.spike_scores["is_spike"].any()


def test_station_readings_and_failures() -> None:
    ok = DashboardService(FakeAirQuality(), FakeGeocoder(), extra_providers=[FakeStation()])
    d = ok.dashboard(DELHI, "us_epa")
    assert len(d.stations) == 1
    assert d.stations[0].pm2_5 == 55.0  # last non-null value carried forward
    assert d.stations[0].pm10 is None
    assert "Synthetic station" in d.attribution
    bad = DashboardService(
        FakeAirQuality(), FakeGeocoder(), extra_providers=[FakeStation(fail=True)]
    )
    d = bad.dashboard(DELHI, "us_epa")
    assert d.stations == []
    assert any("station" in w for w in d.warnings)


def test_compare_cities() -> None:
    places = [Location(f"C{i}", float(i), float(i), "UTC") for i in range(4)]
    svc = DashboardService(FakeAirQuality(fail_for={"C2"}), FakeGeocoder())
    result = svc.compare(places, "us_epa", days=7)
    assert len(result.summary) == 3
    assert result.errors == {"C2": "C2 unavailable"}
    assert list(result.summary["aqi"]) == sorted(result.summary["aqi"], reverse=True)
    assert set(result.daily.columns) == {"C0", "C1", "C3"}
    assert len(result.daily) <= 7
    with pytest.raises(ValueError):
        svc.compare(places * 2, "us_epa")
    assert MAX_COMPARE == 5


def test_compare_with_no_data_row() -> None:
    svc = DashboardService(FakeAirQuality(now="2030-01-01T00:00Z"), FakeGeocoder())
    result = svc.compare([DELHI], "us_epa")
    assert result.summary.iloc[0]["category"] == "Insufficient data"
    assert np.isnan(result.summary.iloc[0]["aqi"])


def test_search_delegates() -> None:
    svc = DashboardService(FakeAirQuality(), FakeGeocoder())
    assert [p.name for p in svc.search("del")] == ["Delhi"]


def test_service_with_saved_model(tmp_path: Path, forecaster: Forecaster) -> None:
    forecaster.model.save(tmp_path)
    svc = DashboardService(
        FakeAirQuality(),
        FakeGeocoder(),
        weather=FakeWeather(),
        forecaster=Forecaster.load(tmp_path),
    )
    assert svc.dashboard(DELHI, "in_naqi").forecast.model_available
