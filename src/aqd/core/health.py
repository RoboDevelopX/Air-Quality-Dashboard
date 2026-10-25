"""Health guidance per AQI category.

US EPA wording follows AirNow's category descriptions and activity guidance; NAQI wording
follows CPCB's "associated health impacts" table. Shown in the app as general information,
not medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass

from aqd.core.categories import AqiStandard, Category

SENSITIVE_GROUPS = (
    "People with heart or lung disease (including asthma), older adults, children and "
    "teenagers, pregnant people, and people who work or exercise outdoors."
)


@dataclass(frozen=True, slots=True)
class HealthAdvice:
    summary: str
    sensitive_groups: str
    everyone: str


_US_EPA: dict[str, HealthAdvice] = {
    "good": HealthAdvice(
        "Air quality is satisfactory, and air pollution poses little or no risk.",
        "No precautions needed. It's a good day to be active outside.",
        "No precautions needed. It's a good day to be active outside.",
    ),
    "moderate": HealthAdvice(
        "Air quality is acceptable, but there may be a risk for people who are unusually "
        "sensitive to air pollution.",
        "If you are unusually sensitive, consider making long or intense outdoor "
        "activities shorter and less intense. Watch for coughing or shortness of breath.",
        "No precautions needed for most people.",
    ),
    "usg": HealthAdvice(
        "Members of sensitive groups may experience health effects. The general public is "
        "less likely to be affected.",
        "Make long or intense outdoor activities shorter and less intense, and take more "
        "breaks. Keep quick-relief medicine handy if you have asthma.",
        "It's fine to be active outside; watch for symptoms such as coughing.",
    ),
    "unhealthy": HealthAdvice(
        "Some members of the general public may experience health effects; sensitive "
        "groups may experience more serious effects.",
        "Avoid long or intense outdoor activities. Move activities indoors or reschedule.",
        "Reduce long or intense outdoor activities and take more breaks.",
    ),
    "very_unhealthy": HealthAdvice(
        "Health alert: the risk of health effects is increased for everyone.",
        "Avoid all physical activity outdoors. Keep indoor air clean, for example with "
        "windows closed and an air purifier if available.",
        "Avoid long or intense outdoor activities; consider moving them indoors.",
    ),
    "hazardous": HealthAdvice(
        "Health warning of emergency conditions: everyone is more likely to be affected.",
        "Remain indoors and keep activity levels low. Follow advice from local authorities.",
        "Avoid all physical activity outdoors. Follow advice from local authorities.",
    ),
}

_IN_NAQI: dict[str, HealthAdvice] = {
    "good": HealthAdvice(
        "Minimal impact.",
        "No precautions needed.",
        "No precautions needed.",
    ),
    "satisfactory": HealthAdvice(
        "May cause minor breathing discomfort to sensitive people.",
        "Consider reducing prolonged or heavy outdoor exertion if you notice symptoms.",
        "No precautions needed for most people.",
    ),
    "moderate": HealthAdvice(
        "May cause breathing discomfort to people with lung disease such as asthma, and "
        "discomfort to people with heart disease, children and older adults.",
        "Reduce prolonged or heavy outdoor exertion. Keep medication handy.",
        "Unusually sensitive people should limit prolonged outdoor exertion.",
    ),
    "poor": HealthAdvice(
        "May cause breathing discomfort to most people on prolonged exposure, and "
        "discomfort to people with heart disease.",
        "Avoid prolonged or heavy outdoor exertion; prefer indoor activities.",
        "Reduce prolonged or heavy outdoor exertion.",
    ),
    "very_poor": HealthAdvice(
        "May cause respiratory illness on prolonged exposure; effects may be more "
        "pronounced in people with lung and heart diseases.",
        "Avoid outdoor physical activity. Keep indoor air clean.",
        "Avoid prolonged or heavy outdoor exertion.",
    ),
    "severe": HealthAdvice(
        "May cause respiratory effects even in healthy people, and serious health impacts "
        "for people with lung or heart disease, even during light physical activity.",
        "Remain indoors and keep activity levels low. Follow advice from local authorities.",
        "Avoid all outdoor physical activity. Follow advice from local authorities.",
    ),
}

_ADVICE: dict[AqiStandard, dict[str, HealthAdvice]] = {"us_epa": _US_EPA, "in_naqi": _IN_NAQI}

DISCLAIMER = (
    "General guidance from the US EPA (AirNow) and India's CPCB, not medical advice. "
    "Follow local authorities and your doctor."
)


def advice_for(category: Category, standard: AqiStandard) -> HealthAdvice:
    return _ADVICE[standard][category.key]
