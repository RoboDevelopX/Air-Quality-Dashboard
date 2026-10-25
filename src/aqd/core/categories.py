"""AQI categories for each standard, with accessible display attributes.

Colour is never the only signal: every category also has a text label and a level
number ("3 of 6"), and the text colour drawn on the category colour is chosen to meet
WCAG AA contrast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AqiStandard = Literal["us_epa", "in_naqi"]


@dataclass(frozen=True, slots=True)
class Category:
    key: str
    label: str
    level: int  # 1 = best
    aqi_lo: int
    aqi_hi: int
    color: str  # background hex, the standard's official (or conventional) colour

    @property
    def text_color(self) -> str:
        """Black or white, whichever contrasts more with `color`."""
        return "#000000" if contrast_ratio(self.color, "#000000") >= 4.5 else "#FFFFFF"


# AirNow's official AQI colours.
US_EPA_CATEGORIES: tuple[Category, ...] = (
    Category("good", "Good", 1, 0, 50, "#00E400"),
    Category("moderate", "Moderate", 2, 51, 100, "#FFFF00"),
    Category("usg", "Unhealthy for Sensitive Groups", 3, 101, 150, "#FF7E00"),
    Category("unhealthy", "Unhealthy", 4, 151, 200, "#FF0000"),
    Category("very_unhealthy", "Very Unhealthy", 5, 201, 300, "#8F3F97"),
    Category("hazardous", "Hazardous", 6, 301, 500, "#7E0023"),
)

# Colours follow CPCB's published bulletin palette (dark green to maroon).
IN_NAQI_CATEGORIES: tuple[Category, ...] = (
    Category("good", "Good", 1, 0, 50, "#009933"),
    Category("satisfactory", "Satisfactory", 2, 51, 100, "#7CBA3B"),
    Category("moderate", "Moderately Polluted", 3, 101, 200, "#FFFF00"),
    Category("poor", "Poor", 4, 201, 300, "#FF9900"),
    Category("very_poor", "Very Poor", 5, 301, 400, "#FF0000"),
    Category("severe", "Severe", 6, 401, 500, "#990000"),
)

CATEGORIES: dict[AqiStandard, tuple[Category, ...]] = {
    "us_epa": US_EPA_CATEGORIES,
    "in_naqi": IN_NAQI_CATEGORIES,
}

STANDARD_NAMES: dict[AqiStandard, str] = {
    "us_epa": "US EPA AQI",
    "in_naqi": "India NAQI",
}


def categorize(aqi: float, standard: AqiStandard) -> Category:
    """Category for an AQI value. Values above the scale map to the top category."""
    if aqi < 0:
        raise ValueError(f"AQI cannot be negative: {aqi}")
    cats = CATEGORIES[standard]
    rounded = round(aqi)
    for cat in cats:
        if rounded <= cat.aqi_hi:
            return cat
    return cats[-1]


def _relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    r, g, b = linear
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2.x contrast ratio between two hex colours (1 to 21)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
