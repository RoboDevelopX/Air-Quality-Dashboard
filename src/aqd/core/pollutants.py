"""Pollutant identifiers and unit conversion.

Concentrations travel through the app in µg/m³ (what Open-Meteo returns). Only the AQI
code converts gases to the units each standard's breakpoint table is written in.
"""

from __future__ import annotations

from enum import StrEnum


class Pollutant(StrEnum):
    PM2_5 = "pm2_5"
    PM10 = "pm10"
    NO2 = "no2"
    O3 = "o3"
    SO2 = "so2"
    CO = "co"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    Pollutant.PM2_5: "PM2.5",
    Pollutant.PM10: "PM10",
    Pollutant.NO2: "NO₂",
    Pollutant.O3: "O₃",
    Pollutant.SO2: "SO₂",
    Pollutant.CO: "CO",
}

# Molar volume of an ideal gas at 25 °C and 1 atm (L/mol), the reference conditions
# US EPA uses for ppb <-> µg/m³ conversion.
MOLAR_VOLUME_25C = 24.45

MOLECULAR_WEIGHT = {  # g/mol
    Pollutant.NO2: 46.0055,
    Pollutant.O3: 47.9982,
    Pollutant.SO2: 64.066,
    Pollutant.CO: 28.010,
}


def ugm3_to_ppb(value: float, pollutant: Pollutant) -> float:
    """Convert a gas concentration from µg/m³ to ppb at 25 °C, 1 atm."""
    if pollutant not in MOLECULAR_WEIGHT:
        raise ValueError(f"{pollutant} is not a gas; no ppb conversion")
    return value * MOLAR_VOLUME_25C / MOLECULAR_WEIGHT[pollutant]


def ugm3_to_ppm(value: float, pollutant: Pollutant) -> float:
    return ugm3_to_ppb(value, pollutant) / 1000.0
