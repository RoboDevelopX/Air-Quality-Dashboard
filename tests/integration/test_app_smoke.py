"""Run the real Streamlit page in-process with fake providers (no network)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import aqd.ui.app as ui_app
from aqd.core.service import DashboardService
from aqd.ml.dataset import CityData, stack
from aqd.ml.inference import Forecaster
from aqd.ml.models import LgbmForecaster
from aqd.providers.base import Location
from tests.fakes import FakeAirQuality, FakeGeocoder, FakeWeather
from tests.synthetic import city_frame

APP = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture(scope="module")
def forecaster() -> Forecaster:
    frame = city_frame(days=200, seed=3, level=60)
    train = stack(
        {"X": CityData(Location("X", 0, 0), frame)}, origin_stride_h=6, horizons_per_origin=12
    )
    model = LgbmForecaster.fit(train, num_boost_round=40, params={"num_leaves": 15})
    model.intervals = {h: (-0.3, 0.3) for h in range(1, 49)}
    model.metadata = {
        "summary": {
            "cities": ["X"],
            "data_start": "2023-01-01",
            "data_end": "2023-07-01",
            "cv": {
                "n_folds": 6,
                "interval_coverage_80": 0.8,
                "buckets": {
                    "1-6h": {"mae": 3.0, "skill_vs_persistence": 0.25},
                    "25-48h": {"mae": 9.0, "skill_vs_persistence": -0.05},
                },
            },
        }
    }
    return Forecaster(model)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("AQD_DEFAULT_CITY", "Delhi")
    monkeypatch.setenv("AQD_CACHE_PATH", str(tmp_path / "cache.sqlite3"))
    monkeypatch.setenv("AQD_LOG_FORMAT", "text")
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def use(monkeypatch: pytest.MonkeyPatch, service: DashboardService) -> None:
    monkeypatch.setattr(ui_app, "build_service", lambda settings: service)


def run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += (
        [i.value for i in at.info] + [w.value for w in at.warning] + [e.value for e in at.error]
    )
    return "\n".join(str(p) for p in parts)


def test_full_dashboard_renders(monkeypatch: pytest.MonkeyPatch, forecaster: Forecaster) -> None:
    use(
        monkeypatch,
        DashboardService(
            FakeAirQuality(), FakeGeocoder(), weather=FakeWeather(), forecaster=forecaster
        ),
    )
    at = run()
    assert not at.exception
    text = all_text(at)
    assert "level " in text and "of 6" in text  # headline shows the level, not just colour
    assert "Health guidance" in [h.value for h in at.subheader]
    assert "Highest expected in the next 48 hours" in text
    assert "25% lower error than assuming no change" in text
    assert "25 to 48 hours ahead: no better than assuming no change" in text
    assert len(at.get("plotly_chart")) >= 4  # daily, hourly, pollutants, forecast (+ compare)
    assert [t.label for t in at.tabs] == ["Now", "History", "Forecast", "Compare cities", "About"]


def test_switching_standard_and_period(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, DashboardService(FakeAirQuality(), FakeGeocoder()))
    at = run()
    at.selectbox[0].select("in_naqi").run()
    assert not at.exception
    assert "India NAQI" in all_text(at)
    at.radio[0].set_value(7).run()
    assert not at.exception


def test_no_model_message(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, DashboardService(FakeAirQuality(), FakeGeocoder()))
    at = run()
    assert not at.exception
    assert "model isn't available" in all_text(at)


def test_unknown_city(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, DashboardService(FakeAirQuality(), FakeGeocoder()))
    at = run()
    at.text_input[0].set_value("Atlantis").run()
    assert not at.exception
    assert "No places match" in all_text(at)


def test_provider_error_shows_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, DashboardService(FakeAirQuality(fail_for={"Delhi"}), FakeGeocoder()))
    at = run()
    assert not at.exception
    assert "Couldn't load air quality data" in all_text(at)
    assert any(b.label == "Try again" for b in at.button)


def test_compare_add_city(monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, DashboardService(FakeAirQuality(), FakeGeocoder()))
    at = run()
    at.text_input(key="compare_query").set_value("London")
    next(b for b in at.button if b.label == "Add").click()
    at.run()
    assert not at.exception
    cities = at.session_state["compare_cities"]
    assert [c.name for c in cities] == ["Delhi", "London"]
