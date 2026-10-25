import numpy as np
import pandas as pd
import pytest

from aqd.core.aqi_series import averaged_inputs, daily_aqi, hourly_aqi, latest_valid


def frame(hours: int, start: str = "2025-01-01", **cols: float) -> pd.DataFrame:
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({k: [v] * hours for k, v in cols.items()}, index=idx)


ALL = {"pm2_5": 12.0, "pm10": 40.0, "no2": 20.0, "o3": 60.0, "so2": 5.0, "co": 300.0}


def test_hourly_epa_constant_levels() -> None:
    out = hourly_aqi(frame(48, **ALL), "us_epa")
    last = out.iloc[-1]
    assert last["aqi"] == 56  # PM2.5 12.0 -> 56 dominates
    assert last["dominant"] == "pm2_5"
    assert last["category"] == "moderate"
    assert last["sub_pm10"] == 37


def test_hourly_naqi_needs_16_hours() -> None:
    out = hourly_aqi(frame(30, **ALL), "in_naqi")
    assert out["aqi"].iloc[:15].isna().all()
    assert out["aqi"].iloc[15:].notna().all()
    assert out["category"].iloc[0] == ""


def test_hourly_missing_columns_are_tolerated() -> None:
    out = hourly_aqi(frame(10, pm2_5=40.0), "us_epa")
    assert out["aqi"].iloc[-1] == 112
    assert np.isnan(out["sub_no2"].iloc[-1])


def test_hourly_empty_frame() -> None:
    assert hourly_aqi(pd.DataFrame(), "us_epa").empty


def test_averaged_inputs_columns_per_standard() -> None:
    epa = averaged_inputs(frame(24, **ALL), "us_epa")
    naqi = averaged_inputs(frame(24, **ALL), "in_naqi")
    assert epa["no2_1h"].notna().all()
    assert epa["no2_24h"].isna().all()
    assert naqi["no2_24h"].iloc[-1] == 20.0
    assert naqi["o3_1h"].isna().all()


def test_daily_aqi_groups_by_local_day() -> None:
    # 72 hours starting at local midnight in Tokyo (UTC+9) -> exactly 3 full local days.
    start = pd.Timestamp("2025-01-01", tz="Asia/Tokyo").tz_convert("UTC")
    df = frame(72, start=str(start), **ALL)
    df.loc[df.index[24:48], "pm2_5"] = 40.0  # second local day is worse
    out = daily_aqi(df, "us_epa", tz="Asia/Tokyo")
    assert len(out) == 3
    assert out["aqi"].tolist() == [56, 112, 56]
    assert str(out.index[1]) == "2025-01-02"


def test_daily_aqi_incomplete_day_is_nan() -> None:
    out = daily_aqi(frame(30, pm2_5=12.0, no2=20.0), "us_epa", tz="UTC")
    assert out["aqi"].iloc[0] == 56
    assert np.isnan(out["aqi"].iloc[1])  # only 6 hours on day two, not even NO2 counts


def test_daily_naqi() -> None:
    out = daily_aqi(frame(48, **ALL), "in_naqi", tz="UTC")
    assert out["aqi"].notna().all()
    assert set(out["category"]) <= {"good", "satisfactory"}


def test_daily_empty() -> None:
    assert daily_aqi(pd.DataFrame(), "in_naqi").empty


def test_latest_valid() -> None:
    out = hourly_aqi(frame(30, **ALL), "in_naqi")
    row = latest_valid(out)
    assert row is not None
    assert row.name == out.index[-1]
    assert latest_valid(out.iloc[:5]) is None


@pytest.mark.parametrize("standard", ["us_epa", "in_naqi"])
def test_negative_values_are_treated_as_missing(standard: str) -> None:
    df = frame(30, **ALL)
    df["so2"] = -1.0
    out = hourly_aqi(df, standard)  # type: ignore[arg-type]
    assert np.isnan(out["sub_so2"].iloc[-1])
