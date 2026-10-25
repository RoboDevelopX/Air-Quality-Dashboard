"""AQI breakpoint tests.

US EPA reference values are computed by hand from 40 CFR 58 App. G Table 2 (2024) with the
CFR's truncation and rounding rules; NAQI values from the CPCB 2014 breakpoint table.
"""

import itertools

import pytest

from aqd.core.aqi import (
    IN_NAQI_SCALES,
    US_EPA_SCALES,
    Concentrations,
    InsufficientDataError,
    compute_aqi,
    interpolate,
    pm25_sub_index,
    truncate,
)
from aqd.core.pollutants import MOLAR_VOLUME_25C, MOLECULAR_WEIGHT, Pollutant


def ppb_to_ugm3(ppb: float, p: Pollutant) -> float:
    return ppb * MOLECULAR_WEIGHT[p] / MOLAR_VOLUME_25C


def ppm_to_ugm3(ppm: float, p: Pollutant) -> float:
    return ppb_to_ugm3(ppm * 1000, p)


def epa(**kw: float) -> int:
    return compute_aqi(Concentrations(**kw), "us_epa").value


# --- truncation -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (9.05, 1, 9.0),
        (35.49, 1, 35.4),
        (35.5, 1, 35.5),
        (54.9, 0, 54.0),
        (0.0789, 3, 0.078),
        (0.07799999999999, 3, 0.078),  # float noise from unit conversion is not truncated
        (0.0, 1, 0.0),
    ],
)
def test_truncate(value: float, decimals: int, expected: float) -> None:
    assert truncate(value, decimals) == expected


# --- US EPA PM2.5: every breakpoint edge, both sides ------------------------------------


@pytest.mark.parametrize(
    ("pm25", "expected"),
    [
        (0.0, 0),
        (9.0, 50),
        (9.05, 50),  # truncated to 9.0
        (9.1, 51),
        (12.0, 56),
        (35.4, 100),
        (35.5, 101),
        (35.9, 102),  # worked example in EPA's AQI technical assistance document
        (55.4, 150),
        (55.5, 151),
        (125.4, 200),
        (125.5, 201),
        (225.4, 300),
        (225.5, 301),
        (325.4, 500),
    ],
)
def test_epa_pm25_breakpoints(pm25: float, expected: int) -> None:
    assert epa(pm2_5_24h=pm25) == expected


def test_epa_pm25_above_scale_extrapolates() -> None:
    result = compute_aqi(Concentrations(pm2_5_24h=425.4), "us_epa")
    # (500-301)/(325.4-225.5) * (425.4-225.5) + 301 = 699.2
    assert result.value == 699
    assert result.above_scale
    assert result.category.key == "hazardous"


@pytest.mark.parametrize(
    ("pm10", "expected"),
    [(0, 0), (54, 50), (54.9, 50), (55, 51), (154, 100), (155, 101), (424, 300), (604, 500)],
)
def test_epa_pm10_breakpoints(pm10: float, expected: int) -> None:
    assert epa(pm10_24h=pm10) == expected


@pytest.mark.parametrize(
    ("ppm", "expected"),
    [(0.054, 50), (0.055, 51), (0.070, 100), (0.071, 101), (0.078, 126), (0.200, 300)],
)
def test_epa_ozone_8h(ppm: float, expected: int) -> None:
    assert epa(o3_8h=ppm_to_ugm3(ppm, Pollutant.O3)) == expected


def test_epa_ozone_uses_higher_of_1h_and_8h() -> None:
    # 8h 0.050 ppm -> 46; 1h 0.130 ppm -> 107. The more precautionary value wins.
    value = epa(o3_8h=ppm_to_ugm3(0.050, Pollutant.O3), o3_1h=ppm_to_ugm3(0.130, Pollutant.O3))
    assert value == 107


def test_epa_ozone_1h_below_its_scale_is_ignored() -> None:
    value = epa(o3_8h=ppm_to_ugm3(0.050, Pollutant.O3), o3_1h=ppm_to_ugm3(0.100, Pollutant.O3))
    assert value == 46


def test_epa_ozone_above_8h_scale_uses_1h() -> None:
    value = epa(o3_8h=ppm_to_ugm3(0.250, Pollutant.O3), o3_1h=ppm_to_ugm3(0.405, Pollutant.O3))
    assert value == 301


def test_epa_ozone_above_8h_scale_without_1h_extrapolates() -> None:
    result = compute_aqi(Concentrations(o3_8h=ppm_to_ugm3(0.250, Pollutant.O3)), "us_epa")
    assert result.value == 353
    assert result.above_scale


@pytest.mark.parametrize(("ppm", "expected"), [(4.4, 50), (4.5, 51), (9.5, 101), (50.4, 500)])
def test_epa_co_8h(ppm: float, expected: int) -> None:
    assert epa(co_8h=ppm_to_ugm3(ppm, Pollutant.CO)) == expected


@pytest.mark.parametrize(("ppb", "expected"), [(53, 50), (54, 51), (100, 100), (101, 101)])
def test_epa_no2_1h(ppb: float, expected: int) -> None:
    assert epa(no2_1h=ppb_to_ugm3(ppb, Pollutant.NO2)) == expected


@pytest.mark.parametrize(("ppb", "expected"), [(35, 50), (36, 51), (75, 100), (304, 200)])
def test_epa_so2_1h(ppb: float, expected: int) -> None:
    assert epa(so2_1h=ppb_to_ugm3(ppb, Pollutant.SO2)) == expected


def test_epa_so2_above_1h_scale_uses_24h() -> None:
    value = epa(so2_1h=ppb_to_ugm3(500, Pollutant.SO2), so2_24h=ppb_to_ugm3(400, Pollutant.SO2))
    # (300-201)/(604-305) * (400-305) + 201 = 232.45
    assert value == 232


@pytest.mark.parametrize("so2_24h_ppb", [None, 200.0])
def test_epa_so2_above_1h_scale_without_high_24h_is_200(so2_24h_ppb: float | None) -> None:
    so2_24h = None if so2_24h_ppb is None else ppb_to_ugm3(so2_24h_ppb, Pollutant.SO2)
    assert epa(so2_1h=ppb_to_ugm3(500, Pollutant.SO2), so2_24h=so2_24h) == 200


def test_epa_overall_is_max_and_reports_dominant() -> None:
    result = compute_aqi(
        Concentrations(pm2_5_24h=35.9, pm10_24h=54, no2_1h=ppb_to_ugm3(54, Pollutant.NO2)),
        "us_epa",
    )
    assert result.value == 102
    assert result.dominant is Pollutant.PM2_5
    assert result.category.key == "usg"
    assert set(result.sub_indices) == {Pollutant.PM2_5, Pollutant.PM10, Pollutant.NO2}
    assert result.sub_indices[Pollutant.NO2].unit == "ppb"


def test_epa_with_no_data_raises() -> None:
    with pytest.raises(InsufficientDataError):
        compute_aqi(Concentrations(), "us_epa")


# --- India NAQI -----------------------------------------------------------------------


def naqi(**kw: float) -> int:
    # NAQI needs >= 3 pollutants incl. PM; pad with near-zero gases so one pollutant is tested.
    base = {"no2_24h": 0.0, "so2_24h": 0.0}
    base.update(kw)
    return compute_aqi(Concentrations(**base), "in_naqi").value


@pytest.mark.parametrize(
    ("pm25", "expected"),
    [
        (0, 0),
        (15, 25),
        (30, 50),
        (45, 75),
        (60, 100),
        (75, 150),
        (90, 200),
        (120, 300),
        (185, 350),
        (250, 400),
        (380, 500),
    ],
)
def test_naqi_pm25(pm25: float, expected: int) -> None:
    assert naqi(pm2_5_24h=pm25) == expected


@pytest.mark.parametrize(
    ("pm10", "expected"), [(50, 50), (100, 100), (175, 150), (250, 200), (350, 300), (430, 400)]
)
def test_naqi_pm10(pm10: float, expected: int) -> None:
    assert naqi(pm10_24h=pm10) == expected


def test_naqi_caps_at_500() -> None:
    result = compute_aqi(Concentrations(pm2_5_24h=600, no2_24h=10, so2_24h=10), "in_naqi")
    assert result.value == 500
    assert result.above_scale
    assert result.category.key == "severe"


def test_naqi_gases() -> None:
    # CO is supplied in µg/m³ and converted to mg/m³: 1500 µg/m³ = 1.5 mg/m³ -> 75.
    result = compute_aqi(
        Concentrations(pm10_24h=40, co_8h=1500, o3_8h=134, no2_24h=200, so2_24h=60),
        "in_naqi",
    )
    subs = result.sub_indices
    assert subs[Pollutant.CO].value == 75
    assert subs[Pollutant.CO].unit == "mg/m³"
    assert subs[Pollutant.O3].value == 150
    assert subs[Pollutant.NO2].value == 220
    assert subs[Pollutant.SO2].value == 75
    assert result.value == 220
    assert result.dominant is Pollutant.NO2
    assert result.category.key == "poor"


@pytest.mark.parametrize(
    "conc",
    [
        Concentrations(pm2_5_24h=50, no2_24h=10),  # only two pollutants
        Concentrations(no2_24h=10, so2_24h=10, o3_8h=10, co_8h=100),  # no PM
    ],
)
def test_naqi_insufficient_data(conc: Concentrations) -> None:
    with pytest.raises(InsufficientDataError):
        compute_aqi(conc, "in_naqi")


# --- shared helpers ---------------------------------------------------------------------


def test_pm25_sub_index_both_standards() -> None:
    assert pm25_sub_index(35.9, "us_epa") == 102
    assert pm25_sub_index(45, "in_naqi") == 75
    assert pm25_sub_index(1000, "in_naqi") == 500


def test_negative_concentration_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        interpolate(-1.0, US_EPA_SCALES["pm2_5_24h"], cap_at_top=False)


@pytest.mark.parametrize("scales", [US_EPA_SCALES, IN_NAQI_SCALES])
def test_tables_are_monotonic(scales: dict[str, object]) -> None:
    for scale in scales.values():
        bands = scale.bands  # type: ignore[attr-defined]
        for lo, hi in itertools.pairwise(bands):
            assert hi.c_lo >= lo.c_hi
            assert hi.i_lo >= lo.i_hi
        for b in bands:
            assert b.c_hi > b.c_lo
            assert b.i_hi > b.i_lo
