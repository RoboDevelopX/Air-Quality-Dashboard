import numpy as np
import pandas as pd
import pytest

from aqd.core.averaging import nowcast, rolling_mean, to_hourly_grid


def hourly(values: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_to_hourly_grid_inserts_missing_hours() -> None:
    s = hourly([1.0, 2.0, 3.0, 4.0])
    gappy = s.drop(s.index[1:3]).to_frame("pm2_5")
    grid = to_hourly_grid(gappy)
    assert len(grid) == 4
    assert grid["pm2_5"].isna().tolist() == [False, True, True, False]


def test_to_hourly_grid_merges_sub_hourly_rows() -> None:
    idx = pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"], tz="UTC")
    grid = to_hourly_grid(pd.DataFrame({"x": [1.0, 3.0, 5.0]}, index=idx))
    assert grid["x"].tolist() == [2.0, 5.0]


def test_to_hourly_grid_rejects_non_datetime_index() -> None:
    with pytest.raises(TypeError):
        to_hourly_grid(pd.DataFrame({"x": [1.0]}))


def test_to_hourly_grid_empty_passthrough() -> None:
    assert to_hourly_grid(pd.DataFrame()).empty


def test_rolling_mean_respects_min_valid() -> None:
    s = hourly([1.0] * 5 + [np.nan] * 3 + [3.0] * 8)
    r = rolling_mean(s, 8, 6)
    assert np.isnan(r.iloc[4])  # only 5 values so far
    assert np.isnan(r.iloc[10])  # window has 3 NaNs -> 5 valid
    assert r.iloc[-1] == 3.0


def test_nowcast_constant_series_equals_value() -> None:
    out = nowcast(hourly([20.0] * 24))
    assert np.isnan(out.iloc[0])  # a single hour is not enough
    assert np.allclose(out.iloc[1:], 20.0)


def test_nowcast_weight_floor() -> None:
    # Most recent 10, previous 40: w = 0.25 -> floored to 0.5.
    # NowCast = (10 + 0.5 * 40) / (1 + 0.5) = 20
    out = nowcast(hourly([40.0, 10.0]))
    assert out.iloc[-1] == pytest.approx(20.0)


def test_nowcast_uses_ratio_when_above_floor() -> None:
    # Values 30 (older) and 40 (current): w = 0.75.
    out = nowcast(hourly([30.0, 40.0]))
    assert out.iloc[-1] == pytest.approx((40 + 0.75 * 30) / 1.75)


def test_nowcast_needs_two_of_last_three_hours() -> None:
    out = nowcast(hourly([10.0, 10.0, 10.0, np.nan, np.nan, 10.0]))
    assert np.isnan(out.iloc[-1])  # only 1 of the last 3 hours is valid
    assert np.isnan(out.iloc[0])  # first hour: only 1 value exists
    assert not np.isnan(out.iloc[2])


def test_nowcast_all_zero() -> None:
    assert nowcast(hourly([0.0, 0.0, 0.0])).iloc[-1] == 0.0


def test_nowcast_tracks_rise_faster_than_24h_mean() -> None:
    s = hourly([10.0] * 24 + [100.0] * 6)
    nc = nowcast(s).iloc[-1]
    mean24 = s.iloc[-24:].mean()
    assert nc > mean24
