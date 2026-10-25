from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqd.ml.cv import make_folds
from aqd.ml.dataset import CityData, fetch_city, stack
from aqd.ml.evaluate import compare_with_provider, plot_error_by_horizon, plot_predictions, run_cv
from aqd.ml.inference import Forecaster, ForecastUnavailableError
from aqd.ml.metrics import mae
from aqd.ml.models import META_FILE, LgbmForecaster, sarimax_forecast
from aqd.providers.base import Location
from tests.synthetic import city_frame

FAST = {"num_leaves": 15, "min_data_in_leaf": 50, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def cities() -> dict[str, CityData]:
    return {
        "A": CityData(Location("A", 0, 0, "UTC"), city_frame(days=240, seed=1, level=30)),
        "B": CityData(Location("B", 1, 1, "Asia/Kolkata"), city_frame(days=240, seed=2, level=90)),
    }


@pytest.fixture(scope="module")
def frames(cities: dict[str, CityData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = stack(cities, origin_stride_h=4, horizons_per_origin=12, seed=0)
    evaluation = stack(cities, origin_stride_h=12)
    return train, evaluation


@pytest.fixture(scope="module")
def cv_result(cities, frames):  # type: ignore[no-untyped-def]
    train, evaluation = frames
    folds = make_folds(evaluation["origin"], n_folds=3, test_days=14, min_train_days=150)
    return run_cv(
        train, evaluation, folds, cities, sarimax_every_days=7, lgbm_rounds=150, lgbm_params=FAST
    )


def test_stack_subsamples_horizons(frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, evaluation = frames
    assert set(train["city"]) == {"A", "B"}
    per_origin = train.groupby(["city", "origin"]).size().mean()
    assert 8 < per_origin < 16  # ~12 of 48 horizons kept
    assert evaluation.groupby(["city", "origin"]).size().max() == 48
    assert train["pm_log_0"].dtype == np.float32


def test_stack_empty() -> None:
    assert stack({}, origin_stride_h=4).empty


def test_lightgbm_beats_baselines_on_held_out_folds(cv_result) -> None:  # type: ignore[no-untyped-def]
    scores = cv_result.scores(by=None).set_index("model")
    assert scores.loc["LightGBM", "mae"] < scores.loc["persistence", "mae"]
    assert scores.loc["LightGBM", "mae"] < scores.loc["seasonal naive", "mae"]
    assert scores.loc["LightGBM", "skill_vs_persistence"] > 0
    by_bucket = cv_result.scores()
    assert set(by_bucket["group"]) == {"1-6h", "7-24h", "25-48h", "all"}


def test_cv_intervals_are_out_of_sample_and_roughly_calibrated(cv_result) -> None:  # type: ignore[no-untyped-def]
    p = cv_result.predictions
    assert p.loc[p["fold"] == 1, "lgbm_lo"].isna().all()  # nothing to calibrate on yet
    assert p.loc[p["fold"] > 1, "lgbm_lo"].notna().any()
    assert 0.6 < cv_result.interval_coverage() < 0.95


def test_sarimax_evaluated_on_daily_subset(cv_result) -> None:  # type: ignore[no-untyped-def]
    p = cv_result.predictions
    with_sarimax = p[p["sarimax"].notna()]
    assert not with_sarimax.empty
    assert (pd.DatetimeIndex(with_sarimax["origin"]).hour == 0).all()
    table = cv_result.scores_with_sarimax(by=None)
    assert set(table["model"]) == {"persistence", "seasonal naive", "SARIMAX", "LightGBM"}
    assert set(cv_result.scores_by_city()["group"]) == {"A", "B", "all"}


def test_plots_are_written(cv_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    plot_predictions(cv_result, "A", 24, tmp_path / "pred.png")
    plot_error_by_horizon(cv_result, tmp_path / "err.png")
    assert (tmp_path / "pred.png").stat().st_size > 1000
    assert (tmp_path / "err.png").stat().st_size > 1000


def test_save_load_roundtrip_and_inference(
    frames: tuple[pd.DataFrame, pd.DataFrame], cities: dict[str, CityData], tmp_path: Path
) -> None:
    train, evaluation = frames
    model = LgbmForecaster.fit(train, num_boost_round=80, params=FAST)
    model.calibrate(
        evaluation.dropna(subset=["y"]), model.predict_delta(evaluation.dropna(subset=["y"]))
    )
    model.metadata = {"trained_at": "test"}
    model.save(tmp_path)
    loaded = LgbmForecaster.load(tmp_path)
    np.testing.assert_allclose(loaded.predict(evaluation), model.predict(evaluation))
    assert loaded.intervals == model.intervals
    assert loaded.metadata["trained_at"] == "test"

    hourly = cities["B"].hourly
    now = hourly.index[-100]
    history = hourly.loc[:now, ["pm2_5", "pm10", "no2", "o3", "so2", "co"]]
    weather = hourly.drop(columns=["pm2_5", "pm10", "no2", "o3", "so2", "co"])
    fc = Forecaster.load(tmp_path).forecast(history, weather, "Asia/Kolkata")
    assert fc.origin == now
    assert len(fc.frame) == 48
    assert fc.frame.index[0] == now + pd.Timedelta(hours=1)
    assert (fc.frame["lo"] <= fc.frame["pm2_5"]).all() and (
        fc.frame["pm2_5"] <= fc.frame["hi"]
    ).all()
    actual = hourly["pm2_5"].reindex(fc.frame.index)
    persistence = np.full(48, history["pm2_5"].iloc[-1])
    # One origin is noisy, so only check the forecast is in a sane range, not better.
    assert (
        mae(actual.to_numpy(), fc.frame["pm2_5"].to_numpy())
        < 3 * mae(actual.to_numpy(), persistence) + 5
    )

    # Without weather the model still runs (LightGBM handles missing values).
    assert len(Forecaster(loaded).forecast(history, None, "UTC").frame) == 48


def test_load_errors(tmp_path: Path, frames: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    with pytest.raises(FileNotFoundError):
        LgbmForecaster.load(tmp_path)
    model = LgbmForecaster.fit(frames[0], num_boost_round=5, params=FAST, early_stopping_frac=0)
    model.save(tmp_path)
    meta = (tmp_path / META_FILE).read_text().replace('"pm_log_0"', '"renamed"')
    (tmp_path / META_FILE).write_text(meta)
    with pytest.raises(ValueError, match="features"):
        LgbmForecaster.load(tmp_path)


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError):
        LgbmForecaster.fit(pd.DataFrame({"y": [], "pm_log_0": [], "origin": []}))


def test_forecast_needs_history(cities: dict[str, CityData]) -> None:
    model = LgbmForecaster.fit(
        stack(cities, origin_stride_h=24, horizons_per_origin=4), num_boost_round=5, params=FAST
    )
    short = cities["A"].hourly.iloc[:10]
    with pytest.raises(ForecastUnavailableError):
        Forecaster(model).forecast(short, None, "UTC")
    with pytest.raises(ForecastUnavailableError):
        Forecaster(model).forecast(pd.DataFrame(index=short.index), None, "UTC")


def test_sarimax_forecast_shape_and_fallbacks(cities: dict[str, CityData]) -> None:
    pm = cities["A"].hourly["pm2_5"].iloc[: 30 * 24]
    fc = sarimax_forecast(pm, steps=48)
    assert fc.shape == (48,) and np.isfinite(fc).all() and (fc >= 0).all()
    short = pm.iloc[:48]
    assert (sarimax_forecast(short, steps=5) == short.iloc[-1]).all()


def test_compare_with_provider() -> None:
    issued = pd.Timestamp("2025-01-01T00:00Z")
    valid = pd.date_range(issued + pd.Timedelta(hours=1), periods=6, freq="h")
    actual = pd.Series(10.0, index=valid)
    snaps = pd.DataFrame({"issued_at": issued, "valid_time": valid, "pm2_5": 14.0})
    ours = pd.DataFrame({"issued_at": issued, "valid_time": valid, "pm2_5": 11.0})
    table = compare_with_provider(snaps, actual, ours)
    overall = table[table["group"] == "all"].set_index("model")
    assert overall.loc["provider (CAMS)", "mae"] == pytest.approx(4.0)
    assert overall.loc["LightGBM", "mae"] == pytest.approx(1.0)
    assert overall.loc["LightGBM", "skill_vs_provider (CAMS)"] == pytest.approx(0.75)


class _FakeHistory:
    def air_quality_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        return city_frame(days=3)[["pm2_5", "pm10"]]

    def weather_range(self, location: Location, start: date, end: date) -> pd.DataFrame:
        return city_frame(days=2)[["wind_speed_10m"]]

    def weather(self, location: Location, *, past_days: int, forecast_days: int) -> pd.DataFrame:
        raise NotImplementedError


def test_fetch_city_joins_weather() -> None:
    fake = _FakeHistory()
    data = fetch_city(fake, fake, Location("X", 0, 0), date(2023, 1, 1), date(2023, 1, 3))
    assert list(data.hourly.columns) == ["pm2_5", "pm10", "wind_speed_10m"]
    assert data.hourly["wind_speed_10m"].isna().sum() == 24  # third day has no weather
