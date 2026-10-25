"""End to end with a mocked provider: fetch -> train/evaluate -> save -> load -> dashboard."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from aqd.core.service import DashboardService
from aqd.ml.dataset import fetch_city
from aqd.ml.evaluate import score_provider_rows
from aqd.ml.inference import Forecaster
from aqd.ml.train import TrainConfig, provider_comparison, train_pipeline
from aqd.providers.base import Location
from tests.fakes import FakeAirQuality, FakeGeocoder, FakeWeather
from tests.synthetic import city_frame

CITIES = [
    Location("Alpha", 10.0, 10.0, "Asia/Kolkata"),
    Location("Beta", 20.0, 20.0, "Europe/London"),
]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    frames = {
        "Alpha": city_frame(days=260, seed=11, level=80),
        "Beta": city_frame(days=260, seed=12, level=15),
    }
    cities = {}
    for loc in CITIES:
        provider = FakeAirQuality(frames[loc.name])
        weather = FakeWeather(frames[loc.name])
        cities[loc.name] = fetch_city(provider, weather, loc, date(2023, 1, 1), date(2023, 9, 1))

    config = TrainConfig(
        n_folds=2,
        test_days=14,
        min_train_days=150,
        sarimax_every_days=7,
        lgbm_rounds=100,
        lgbm_params={"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": 0.1},
    )
    report = train_pipeline(
        cities, model_dir=tmp_path / "models", report_dir=tmp_path / "reports", config=config
    )

    # Reports and artifact exist and say the model beats persistence on held-out data.
    assert (tmp_path / "models" / "pm25_lgbm.txt").is_file()
    results = (tmp_path / "reports" / "results.md").read_text()
    assert "LightGBM" in results and "SARIMAX" in results
    assert (tmp_path / "reports" / "error_by_horizon.png").is_file()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    horizons = {r["group"] for r in metrics["by_horizon"]} - {"all"}
    assert {int(h) for h in horizons} == set(range(1, 49))
    overall = report.summary["cv"]["buckets"]["all"]
    assert overall["skill_vs_persistence"] > 0
    assert 0 < report.summary["cv"]["interval_coverage_80"] <= 1

    # The app path: load the artifact and render a dashboard view model.
    forecaster = Forecaster.load(tmp_path / "models")
    service = DashboardService(
        FakeAirQuality(frames["Alpha"], now="2023-09-01T06:00Z"),
        FakeGeocoder(CITIES),
        weather=FakeWeather(frames["Alpha"]),
        forecaster=forecaster,
    )
    place = service.search("alp")[0]
    dashboard = service.dashboard(place, "in_naqi", history_days=14)
    assert dashboard.current is not None
    assert dashboard.forecast.model_available
    assert dashboard.forecast.skill["all"] == overall["skill_vs_persistence"]
    assert len(dashboard.daily) > 0

    # Provider comparison only uses snapshots issued after the training data ends.
    issued = pd.Timestamp("2023-09-10T00:00Z")
    valid = pd.date_range(issued + pd.Timedelta(hours=1), periods=48, freq="h")
    snaps = pd.DataFrame({"city": "Alpha", "issued_at": issued, "valid_time": valid, "pm2_5": 80.0})
    old = snaps.assign(issued_at=pd.Timestamp("2023-01-15T00:00Z"))
    rows = provider_comparison(
        report.model,
        cities,
        pd.concat([snaps, old]),
        trained_until=pd.Timestamp("2023-09-05T00:00Z"),
    )
    assert set(rows["issued_at"]) == {issued}
    table = score_provider_rows(rows, by=None).set_index("model")
    assert {"provider (CAMS)", "LightGBM"} <= set(table.index)
