import numpy as np
import pandas as pd
import pytest

from aqd.ml.features import (
    LAGS,
    MAX_HORIZON,
    build_frame,
    feature_columns,
    invert_delta,
    origin_features,
    target_delta,
)
from tests.synthetic import city_frame


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return city_frame(days=20)


def test_shape_and_columns(df: pd.DataFrame) -> None:
    origins = df.index[[200, 300]]
    frame = build_frame(df, "UTC", origins=origins)
    assert len(frame) == 2 * MAX_HORIZON
    assert set(frame["horizon"]) == set(range(1, MAX_HORIZON + 1))
    feats = feature_columns(frame)
    assert {"origin", "valid_time", "y", "persistence", "seasonal_naive"}.isdisjoint(feats)
    assert all(f"pm_log_lag{lag}" in feats for lag in LAGS)
    assert "tgt_wind_u" in feats and "tgt_wind_direction_10m" not in feats


def test_target_and_baselines_are_aligned(df: pd.DataFrame) -> None:
    t = df.index[300]
    frame = build_frame(df, "UTC", horizons=(1, 24, 25, 48), origins=pd.DatetimeIndex([t]))
    by_h = frame.set_index("horizon")
    for h in (1, 24, 25, 48):
        assert by_h.loc[h, "valid_time"] == t + pd.Timedelta(hours=h)
        assert by_h.loc[h, "y"] == pytest.approx(df["pm2_5"].loc[t + pd.Timedelta(hours=h)])
        assert by_h.loc[h, "persistence"] == pytest.approx(df["pm2_5"].loc[t])
    # Seasonal naive: same clock hour on the most recent day at or before the origin.
    assert by_h.loc[1, "seasonal_naive"] == pytest.approx(
        df["pm2_5"].loc[t - pd.Timedelta(hours=23)]
    )
    assert by_h.loc[24, "seasonal_naive"] == pytest.approx(df["pm2_5"].loc[t])
    assert by_h.loc[25, "seasonal_naive"] == pytest.approx(
        df["pm2_5"].loc[t - pd.Timedelta(hours=23)]
    )
    assert by_h.loc[48, "seasonal_naive"] == pytest.approx(df["pm2_5"].loc[t])


def test_no_leakage_from_the_future(df: pd.DataFrame) -> None:
    """Every pollutant-derived feature at origin t is unchanged when all pollutant values
    after t are replaced. Only the target and target-time weather may depend on t+h."""
    t = df.index[400]
    origins = pd.DatetimeIndex([t])
    original = build_frame(df, "UTC", origins=origins)
    tampered_df = df.copy()
    future = tampered_df.index > t
    rng = np.random.default_rng(1)
    for col in ("pm2_5", "pm10", "no2", "o3", "so2", "co"):
        tampered_df.loc[future, col] = rng.uniform(0, 1000, future.sum())
    tampered = build_frame(tampered_df, "UTC", origins=origins)

    weather_at_target = [c for c in original.columns if c.startswith("tgt_")]
    must_match = [c for c in feature_columns(original) if c not in weather_at_target]
    pd.testing.assert_frame_equal(original[must_match], tampered[must_match])
    pd.testing.assert_series_equal(original["persistence"], tampered["persistence"])
    pd.testing.assert_series_equal(original["seasonal_naive"], tampered["seasonal_naive"])
    assert not np.allclose(original["y"], tampered["y"])  # sanity: the target did change


def test_origin_features_do_not_use_current_row_for_lags(df: pd.DataFrame) -> None:
    feats = origin_features(df)
    assert feats["pm_log_lag1"].iloc[10] == pytest.approx(np.log1p(df["pm2_5"].iloc[9]))
    assert feats["pm_log_0"].iloc[10] == pytest.approx(np.log1p(df["pm2_5"].iloc[10]))


def test_calendar_uses_local_time(df: pd.DataFrame) -> None:
    t = pd.Timestamp("2023-01-05T18:00Z")  # midnight in Kolkata is 18:30 UTC
    utc = build_frame(df, "UTC", horizons=(1,), origins=pd.DatetimeIndex([t]))
    ist = build_frame(df, "Asia/Kolkata", horizons=(1,), origins=pd.DatetimeIndex([t]))
    # 19:00 UTC == 00:30 IST: hour_cos is ~1 near local midnight.
    assert ist["tgt_hour_cos"].iloc[0] > 0.99
    assert utc["tgt_hour_cos"].iloc[0] < 0.5


def test_targets_beyond_data_are_nan(df: pd.DataFrame) -> None:
    last = pd.DatetimeIndex([df.index[-1]])
    frame = build_frame(df, "UTC", horizons=(1, 2), origins=last)
    assert frame["y"].isna().all()


def test_missing_weather_and_aux_columns_become_nan(df: pd.DataFrame) -> None:
    frame = build_frame(df[["pm2_5"]], "UTC", horizons=(1,), origins=df.index[[100]])
    assert frame["tgt_temperature_2m"].isna().all()
    assert frame["no2_log_0"].isna().all()


def test_empty_input() -> None:
    assert build_frame(pd.DataFrame(), "UTC").empty


def test_target_roundtrip(df: pd.DataFrame) -> None:
    frame = build_frame(df, "UTC", horizons=(6,), origins=df.index[200:210])
    recovered = invert_delta(frame, target_delta(frame).to_numpy())
    np.testing.assert_allclose(recovered, frame["y"], rtol=1e-9)
