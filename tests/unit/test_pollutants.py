import pytest

from aqd.core.aqi import IN_NAQI_SCALES, US_EPA_SCALES, scales_for
from aqd.core.pollutants import Pollutant, ugm3_to_ppb, ugm3_to_ppm


def test_labels() -> None:
    assert Pollutant.PM2_5.label == "PM2.5"
    assert Pollutant.NO2.label == "NO₂"


def test_conversion_matches_epa_reference() -> None:
    # 1 ppb NO2 at 25 °C ≈ 1.88 µg/m³
    assert ugm3_to_ppb(1.8816, Pollutant.NO2) == pytest.approx(1.0, rel=1e-3)
    # 1 ppm CO ≈ 1145 µg/m³
    assert ugm3_to_ppm(1145.6, Pollutant.CO) == pytest.approx(1.0, rel=1e-3)


def test_particulates_cannot_be_converted() -> None:
    with pytest.raises(ValueError, match="not a gas"):
        ugm3_to_ppb(10.0, Pollutant.PM2_5)


def test_scales_for() -> None:
    assert scales_for("us_epa") is US_EPA_SCALES
    assert scales_for("in_naqi") is IN_NAQI_SCALES
