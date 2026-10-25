import itertools

import pytest

from aqd.core.categories import CATEGORIES, categorize, contrast_ratio
from aqd.core.health import advice_for


@pytest.mark.parametrize(
    ("aqi", "standard", "key"),
    [
        (0, "us_epa", "good"),
        (50, "us_epa", "good"),
        (50.4, "us_epa", "good"),
        (51, "us_epa", "moderate"),
        (150, "us_epa", "usg"),
        (301, "us_epa", "hazardous"),
        (699, "us_epa", "hazardous"),
        (100, "in_naqi", "satisfactory"),
        (101, "in_naqi", "moderate"),
        (200, "in_naqi", "moderate"),
        (201, "in_naqi", "poor"),
        (401, "in_naqi", "severe"),
    ],
)
def test_categorize(aqi: float, standard: str, key: str) -> None:
    assert categorize(aqi, standard).key == key  # type: ignore[arg-type]


def test_categorize_negative_rejected() -> None:
    with pytest.raises(ValueError):
        categorize(-1, "us_epa")


def test_contrast_ratio_extremes() -> None:
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0)


@pytest.mark.parametrize("standard", list(CATEGORIES))
def test_categories_are_accessible_and_complete(standard: str) -> None:
    cats = CATEGORIES[standard]  # type: ignore[index]
    assert [c.level for c in cats] == list(range(1, 7))
    for prev, cur in itertools.pairwise(cats):
        assert cur.aqi_lo == prev.aqi_hi + 1
    for cat in cats:
        assert contrast_ratio(cat.color, cat.text_color) >= 4.5, cat
        advice = advice_for(cat, standard)  # type: ignore[arg-type]
        assert advice.summary and advice.sensitive_groups and advice.everyone
