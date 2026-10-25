"""AQI calculation for US EPA and India NAQI from their breakpoint tables.

Sources:
- US EPA: 40 CFR Part 58, Appendix G, Table 2 (as revised February 2024, which lowered the
  PM2.5 "Good" ceiling to 9.0 µg/m³ and merged the two Hazardous bands into 301-500).
- India NAQI: CPCB National Air Quality Index (2014), breakpoint table and rules.

Both standards compute a sub-index per pollutant by linear interpolation inside the band
that contains the concentration, and report the maximum sub-index as the AQI:

    I = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo

This module works on single, already-averaged concentrations. `core.averaging` turns
hourly series into the averages each standard requires.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from aqd.core.categories import AqiStandard, Category, categorize
from aqd.core.pollutants import Pollutant, ugm3_to_ppb, ugm3_to_ppm


@dataclass(frozen=True, slots=True)
class Band:
    c_lo: float
    c_hi: float
    i_lo: int
    i_hi: int


@dataclass(frozen=True, slots=True)
class Scale:
    """Breakpoints for one pollutant at one averaging period, in the standard's unit."""

    unit: str
    averaging_hours: int
    bands: tuple[Band, ...]
    decimals: int | None = None  # EPA truncates before lookup; NAQI does not


def _bands(rows: list[tuple[float, float, int, int]]) -> tuple[Band, ...]:
    return tuple(Band(*row) for row in rows)


# --- US EPA (40 CFR 58 App. G, Table 2, 2024) ------------------------------------------
# The top concentration of each table is the one the CFR assigns to AQI 500 (footnote 4).
US_EPA_SCALES: dict[str, Scale] = {
    "o3_8h": Scale(
        "ppm",
        8,
        _bands(
            [
                (0.000, 0.054, 0, 50),
                (0.055, 0.070, 51, 100),
                (0.071, 0.085, 101, 150),
                (0.086, 0.105, 151, 200),
                (0.106, 0.200, 201, 300),
            ]
        ),
        decimals=3,
    ),
    # 1-hour ozone only defines AQI >= 101 and is required above 300 (footnotes 1-2).
    "o3_1h": Scale(
        "ppm",
        1,
        _bands(
            [
                (0.125, 0.164, 101, 150),
                (0.165, 0.204, 151, 200),
                (0.205, 0.404, 201, 300),
                (0.405, 0.604, 301, 500),
            ]
        ),
        decimals=3,
    ),
    "pm2_5_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0.0, 9.0, 0, 50),
                (9.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 125.4, 151, 200),
                (125.5, 225.4, 201, 300),
                (225.5, 325.4, 301, 500),
            ]
        ),
        decimals=1,
    ),
    "pm10_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0, 54, 0, 50),
                (55, 154, 51, 100),
                (155, 254, 101, 150),
                (255, 354, 151, 200),
                (355, 424, 201, 300),
                (425, 604, 301, 500),
            ]
        ),
        decimals=0,
    ),
    "co_8h": Scale(
        "ppm",
        8,
        _bands(
            [
                (0.0, 4.4, 0, 50),
                (4.5, 9.4, 51, 100),
                (9.5, 12.4, 101, 150),
                (12.5, 15.4, 151, 200),
                (15.5, 30.4, 201, 300),
                (30.5, 50.4, 301, 500),
            ]
        ),
        decimals=1,
    ),
    # 1-hour SO2 does not define AQI >= 201; those use 24-hour SO2 (footnote 3).
    "so2_1h": Scale(
        "ppb",
        1,
        _bands(
            [
                (0, 35, 0, 50),
                (36, 75, 51, 100),
                (76, 185, 101, 150),
                (186, 304, 151, 200),
            ]
        ),
        decimals=0,
    ),
    "so2_24h": Scale(
        "ppb",
        24,
        _bands([(305, 604, 201, 300), (605, 1004, 301, 500)]),
        decimals=0,
    ),
    "no2_1h": Scale(
        "ppb",
        1,
        _bands(
            [
                (0, 53, 0, 50),
                (54, 100, 51, 100),
                (101, 360, 101, 150),
                (361, 649, 151, 200),
                (650, 1249, 201, 300),
                (1250, 2049, 301, 500),
            ]
        ),
        decimals=0,
    ),
}

# --- India NAQI (CPCB 2014) -------------------------------------------------------------
# The published table lists bands like 0-30, 31-60 for integer readings. Real averages are
# continuous, so each band starts where the previous one ends (30 -> 50, 60 -> 100, ...),
# which is how CPCB's own calculator interpolates. The table leaves "Severe" open-ended;
# the AQI-500 concentration used here is the conventional extension of that band, and
# results are capped at 500 because the NAQI scale ends there.
IN_NAQI_SCALES: dict[str, Scale] = {
    "pm10_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0, 50, 0, 50),
                (50, 100, 50, 100),
                (100, 250, 100, 200),
                (250, 350, 200, 300),
                (350, 430, 300, 400),
                (430, 510, 400, 500),
            ]
        ),
    ),
    "pm2_5_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0, 30, 0, 50),
                (30, 60, 50, 100),
                (60, 90, 100, 200),
                (90, 120, 200, 300),
                (120, 250, 300, 400),
                (250, 380, 400, 500),
            ]
        ),
    ),
    "no2_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0, 40, 0, 50),
                (40, 80, 50, 100),
                (80, 180, 100, 200),
                (180, 280, 200, 300),
                (280, 400, 300, 400),
                (400, 520, 400, 500),
            ]
        ),
    ),
    "o3_8h": Scale(
        "µg/m³",
        8,
        _bands(
            [
                (0, 50, 0, 50),
                (50, 100, 50, 100),
                (100, 168, 100, 200),
                (168, 208, 200, 300),
                (208, 748, 300, 400),
                (748, 1000, 400, 500),
            ]
        ),
    ),
    "co_8h": Scale(
        "mg/m³",
        8,
        _bands(
            [
                (0, 1.0, 0, 50),
                (1.0, 2.0, 50, 100),
                (2.0, 10, 100, 200),
                (10, 17, 200, 300),
                (17, 34, 300, 400),
                (34, 51, 400, 500),
            ]
        ),
    ),
    "so2_24h": Scale(
        "µg/m³",
        24,
        _bands(
            [
                (0, 40, 0, 50),
                (40, 80, 50, 100),
                (80, 380, 100, 200),
                (380, 800, 200, 300),
                (800, 1600, 300, 400),
                (1600, 2400, 400, 500),
            ]
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class SubIndex:
    pollutant: Pollutant
    value: int
    concentration: float  # in the scale's unit, after any truncation
    unit: str
    averaging_hours: int
    above_scale: bool = False


@dataclass(frozen=True, slots=True)
class AqiResult:
    standard: AqiStandard
    value: int
    category: Category
    dominant: Pollutant
    sub_indices: dict[Pollutant, SubIndex]

    @property
    def above_scale(self) -> bool:
        return self.sub_indices[self.dominant].above_scale


def truncate(value: float, decimals: int) -> float:
    """Truncate (not round) to `decimals` places, as EPA requires.

    Float noise is rounded away first (at 6 extra places) so that a value like 0.078 that
    arrives as 0.07799999999 after a unit conversion is not cut to 0.077.
    """
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(round(value, decimals + 6))).quantize(q, rounding=ROUND_DOWN))


def _round_half_up(x: float) -> int:
    return int(Decimal(repr(x)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def interpolate(concentration: float, scale: Scale, *, cap_at_top: bool) -> tuple[int, bool, float]:
    """Return (index, above_scale, concentration_used) for a concentration on a scale.

    Values above the top band are extrapolated along the top band (EPA, 40 CFR 58 App. G
    footnote 4) or capped at the top index (`cap_at_top`, NAQI).
    """
    if concentration < 0:
        raise ValueError(f"concentration cannot be negative: {concentration}")
    c = truncate(concentration, scale.decimals) if scale.decimals is not None else concentration
    for band in scale.bands:
        if c <= band.c_hi:
            # c < c_lo only happens for scales that start above zero (e.g. 1-hour O3);
            # callers check that first, so clamp defensively.
            c_in = max(c, band.c_lo)
            idx = (band.i_hi - band.i_lo) / (band.c_hi - band.c_lo) * (c_in - band.c_lo)
            return _round_half_up(idx + band.i_lo), False, c
    top = scale.bands[-1]
    if cap_at_top:
        return top.i_hi, True, c
    idx = (top.i_hi - top.i_lo) / (top.c_hi - top.c_lo) * (c - top.c_lo) + top.i_lo
    return _round_half_up(idx), True, c


def _sub(pollutant: Pollutant, conc: float, scale: Scale, *, cap_at_top: bool = False) -> SubIndex:
    value, above, used = interpolate(conc, scale, cap_at_top=cap_at_top)
    return SubIndex(pollutant, value, used, scale.unit, scale.averaging_hours, above)


@dataclass(frozen=True, slots=True)
class Concentrations:
    """Averaged concentrations in µg/m³, already averaged over the periods each standard uses.

    Fields left as None are treated as unavailable.
    """

    pm2_5_24h: float | None = None
    pm10_24h: float | None = None
    o3_8h: float | None = None
    o3_1h: float | None = None
    co_8h: float | None = None
    no2_1h: float | None = None
    no2_24h: float | None = None
    so2_1h: float | None = None
    so2_24h: float | None = None


def us_epa_sub_indices(c: Concentrations) -> dict[Pollutant, SubIndex]:
    s = US_EPA_SCALES
    out: dict[Pollutant, SubIndex] = {}
    if c.pm2_5_24h is not None:
        out[Pollutant.PM2_5] = _sub(Pollutant.PM2_5, c.pm2_5_24h, s["pm2_5_24h"])
    if c.pm10_24h is not None:
        out[Pollutant.PM10] = _sub(Pollutant.PM10, c.pm10_24h, s["pm10_24h"])
    if c.co_8h is not None:
        out[Pollutant.CO] = _sub(Pollutant.CO, ugm3_to_ppm(c.co_8h, Pollutant.CO), s["co_8h"])
    if c.no2_1h is not None:
        out[Pollutant.NO2] = _sub(Pollutant.NO2, ugm3_to_ppb(c.no2_1h, Pollutant.NO2), s["no2_1h"])

    # Ozone: 8-hour normally; 1-hour too where it is higher (more precautionary), and
    # 1-hour is the only scale above AQI 300.
    o3: list[SubIndex] = []
    ppm8 = ugm3_to_ppm(c.o3_8h, Pollutant.O3) if c.o3_8h is not None else None
    if ppm8 is not None and truncate(ppm8, 3) <= s["o3_8h"].bands[-1].c_hi:
        o3.append(_sub(Pollutant.O3, ppm8, s["o3_8h"]))
    if c.o3_1h is not None:
        ppm1 = ugm3_to_ppm(c.o3_1h, Pollutant.O3)
        if truncate(ppm1, 3) >= s["o3_1h"].bands[0].c_lo:
            o3.append(_sub(Pollutant.O3, ppm1, s["o3_1h"]))
    if not o3 and ppm8 is not None:
        # 8-hour above its scale with no usable 1-hour value: extrapolate the 8-hour scale.
        o3.append(_sub(Pollutant.O3, ppm8, s["o3_8h"]))
    if o3:
        out[Pollutant.O3] = max(o3, key=lambda si: si.value)

    # SO2: 1-hour up to AQI 200; above 304 ppb the 24-hour average decides (footnote 3).
    # If the 24-hour average is below the 305 ppb floor, the AQI stays at 200.
    if c.so2_1h is not None:
        ppb1 = ugm3_to_ppb(c.so2_1h, Pollutant.SO2)
        if truncate(ppb1, 0) <= s["so2_1h"].bands[-1].c_hi:
            out[Pollutant.SO2] = _sub(Pollutant.SO2, ppb1, s["so2_1h"])
        elif (
            c.so2_24h is not None
            and truncate(ugm3_to_ppb(c.so2_24h, Pollutant.SO2), 0) >= s["so2_24h"].bands[0].c_lo
        ):
            out[Pollutant.SO2] = _sub(
                Pollutant.SO2, ugm3_to_ppb(c.so2_24h, Pollutant.SO2), s["so2_24h"]
            )
        else:
            out[Pollutant.SO2] = SubIndex(Pollutant.SO2, 200, truncate(ppb1, 0), "ppb", 1)
    return out


def in_naqi_sub_indices(c: Concentrations) -> dict[Pollutant, SubIndex]:
    s = IN_NAQI_SCALES
    out: dict[Pollutant, SubIndex] = {}
    pairs: list[tuple[Pollutant, float | None, str, float]] = [
        (Pollutant.PM2_5, c.pm2_5_24h, "pm2_5_24h", 1.0),
        (Pollutant.PM10, c.pm10_24h, "pm10_24h", 1.0),
        (Pollutant.NO2, c.no2_24h, "no2_24h", 1.0),
        (Pollutant.O3, c.o3_8h, "o3_8h", 1.0),
        (Pollutant.CO, c.co_8h, "co_8h", 1e-3),  # NAQI expresses CO in mg/m³
        (Pollutant.SO2, c.so2_24h, "so2_24h", 1.0),
    ]
    for pollutant, value, key, factor in pairs:
        if value is not None:
            out[pollutant] = _sub(pollutant, value * factor, s[key], cap_at_top=True)
    return out


class InsufficientDataError(ValueError):
    """Not enough pollutants to report an overall AQI under the standard's rules."""


def compute_aqi(c: Concentrations, standard: AqiStandard) -> AqiResult:
    """Overall AQI = max sub-index. Raises InsufficientDataError when the rules aren't met.

    NAQI requires at least three pollutants, one of them PM2.5 or PM10 (CPCB rule).
    EPA reports from whatever pollutants are available, so one is enough.
    """
    if standard == "us_epa":
        subs = us_epa_sub_indices(c)
        if not subs:
            raise InsufficientDataError("no pollutant concentrations available")
    else:
        subs = in_naqi_sub_indices(c)
        has_pm = Pollutant.PM2_5 in subs or Pollutant.PM10 in subs
        if len(subs) < 3 or not has_pm:
            raise InsufficientDataError(
                "NAQI needs at least 3 pollutants including PM2.5 or PM10; "
                f"got {sorted(p.value for p in subs)}"
            )
    dominant = max(subs, key=lambda p: subs[p].value)
    value = subs[dominant].value
    return AqiResult(standard, value, categorize(value, standard), dominant, subs)


def pm25_sub_index(pm2_5_24h: float, standard: AqiStandard) -> int:
    """PM2.5-only index, used for the forecast (the model forecasts PM2.5 only)."""
    if standard == "us_epa":
        return _sub(Pollutant.PM2_5, pm2_5_24h, US_EPA_SCALES["pm2_5_24h"]).value
    return _sub(Pollutant.PM2_5, pm2_5_24h, IN_NAQI_SCALES["pm2_5_24h"], cap_at_top=True).value


def scales_for(standard: AqiStandard) -> Mapping[str, Scale]:
    return US_EPA_SCALES if standard == "us_epa" else IN_NAQI_SCALES
