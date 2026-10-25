"""Application service: everything the UI shows, as plain view models.

The UI calls `DashboardService` and renders the dataclasses it returns. All decisions
(which averages, which category, whether a forecast is trustworthy enough to show, what to
say when data is missing) are made here so they can be unit tested without Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from aqd.core.anomaly import SpikeEpisode, detect_spikes, episodes
from aqd.core.aqi import Concentrations, InsufficientDataError, compute_aqi, pm25_sub_index
from aqd.core.aqi_series import averaged_inputs, daily_aqi, hourly_aqi
from aqd.core.averaging import MIN_HOURS_24H_NAQI, nowcast, rolling_mean
from aqd.core.categories import STANDARD_NAMES, AqiStandard, Category, categorize
from aqd.core.health import HealthAdvice, advice_for
from aqd.core.pollutants import Pollutant
from aqd.ml.inference import Forecaster, ForecastUnavailableError
from aqd.providers.base import Geocoder, Location, Provider, WeatherProvider
from aqd.providers.http import ProviderError

log = logging.getLogger(__name__)

MAX_COMPARE = 5
FORECAST_HOURS = 48

AVERAGING_LABEL: dict[AqiStandard, dict[Pollutant, str]] = {
    "us_epa": {
        Pollutant.PM2_5: "NowCast (12h weighted)",
        Pollutant.PM10: "NowCast (12h weighted)",
        Pollutant.O3: "8-hour (or 1-hour if higher)",
        Pollutant.CO: "8-hour",
        Pollutant.NO2: "1-hour",
        Pollutant.SO2: "1-hour",
    },
    "in_naqi": {
        Pollutant.PM2_5: "24-hour",
        Pollutant.PM10: "24-hour",
        Pollutant.O3: "8-hour",
        Pollutant.CO: "8-hour",
        Pollutant.NO2: "24-hour",
        Pollutant.SO2: "24-hour",
    },
}


@dataclass(frozen=True, slots=True)
class PollutantReading:
    pollutant: Pollutant
    concentration: float | None  # latest hourly value, µg/m³ (mg/m³ for CO display)
    unit: str
    sub_index: int | None
    averaging: str


@dataclass(frozen=True, slots=True)
class CurrentConditions:
    time: pd.Timestamp  # UTC hour the reading refers to
    aqi: int
    category: Category
    dominant: Pollutant
    above_scale: bool
    advice: HealthAdvice
    pollutants: list[PollutantReading]


@dataclass(frozen=True, slots=True)
class ForecastView:
    """Model and provider PM2.5 forecasts on one hourly index.

    `frame` columns: model, model_lo, model_hi, model_aqi, model_category, provider,
    provider_aqi, observed (recent history for context). Model columns are NaN when the
    model is unavailable, with `message` saying why.
    """

    frame: pd.DataFrame
    model_available: bool
    message: str | None
    origin: pd.Timestamp | None
    skill: dict[str, float]  # CV skill vs persistence per horizon bucket (from training)
    interval_coverage: float | None
    peak: tuple[pd.Timestamp, float, Category] | None  # worst forecast hour (model or provider)


@dataclass(frozen=True, slots=True)
class StationReading:
    """Latest ground-station measurement from an optional provider (e.g. OpenAQ)."""

    source: str
    time: pd.Timestamp
    pm2_5: float | None
    pm10: float | None


@dataclass(frozen=True, slots=True)
class Dashboard:
    location: Location
    standard: AqiStandard
    standard_name: str
    issued_at: datetime
    stale: bool
    current: CurrentConditions | None
    current_message: str | None
    history: pd.DataFrame  # hourly pollutants, µg/m³
    hourly: pd.DataFrame  # hourly AQI, category, dominant, sub-indices
    daily: pd.DataFrame  # daily AQI per local day
    spikes: list[SpikeEpisode]
    spike_scores: pd.DataFrame
    forecast: ForecastView
    attribution: list[str]
    stations: list[StationReading] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CityComparison:
    summary: pd.DataFrame  # one row per city: label, aqi, category, dominant, pm2_5
    daily: pd.DataFrame  # columns = city labels, index = local date, values = daily AQI
    errors: dict[str, str]


class DashboardService:
    def __init__(
        self,
        provider: Provider,
        geocoder: Geocoder,
        *,
        weather: WeatherProvider | None = None,
        forecaster: Forecaster | None = None,
        extra_providers: list[Provider] | None = None,
    ) -> None:
        self.provider = provider
        self.geocoder = geocoder
        self.weather = weather
        self.forecaster = forecaster
        self.extra_providers = extra_providers or []

    def search(self, query: str, count: int = 5) -> list[Location]:
        return self.geocoder.search(query, count)

    def dashboard(
        self, location: Location, standard: AqiStandard, history_days: int = 30
    ) -> Dashboard:
        """Build the full dashboard. Raises ProviderError only if the primary source fails
        with nothing cached; secondary failures become warnings."""
        if history_days not in range(1, 93):
            raise ValueError("history_days must be between 1 and 92")
        # Always fetch at least 30 days: the 7- and 30-day views then share one cached
        # response, and 24h averages, NowCast and the 7-day anomaly scale are warmed up.
        past_days = max(history_days, 30) + 1
        data = self.provider.air_quality(
            location, past_days=past_days, forecast_days=FORECAST_HOURS // 24 + 1
        )
        warnings: list[str] = []
        if data.stale:
            warnings.append(
                "Live data is unavailable, so this shows the most recent saved copy. "
                f"It was fetched at {data.issued_at:%Y-%m-%d %H:%M} UTC."
            )
        history = data.history.dropna(how="all")
        hourly = hourly_aqi(history, standard)
        daily = daily_aqi(history, standard, tz=location.timezone)
        cutoff = history.index.max() - pd.Timedelta(days=history_days) if len(history) else None
        current, current_message = _current(history, standard)
        scores = detect_spikes(history["pm2_5"]) if "pm2_5" in history else pd.DataFrame()
        recent_scores = scores[scores.index > cutoff] if cutoff is not None else scores
        weather = self._weather(location, past_days, warnings)
        forecast = self._forecast(history, data.forecast, weather, location, standard)
        attribution = [self.provider.attribution]
        stations = []
        for extra in self.extra_providers:
            attribution.append(extra.attribution)
            reading = _station(extra, location, warnings)
            if reading is not None:
                stations.append(reading)
        return Dashboard(
            location=location,
            standard=standard,
            standard_name=STANDARD_NAMES[standard],
            issued_at=data.issued_at,
            stale=data.stale,
            current=current,
            current_message=current_message,
            history=history[history.index > cutoff] if cutoff is not None else history,
            hourly=hourly[hourly.index > cutoff] if cutoff is not None else hourly,
            daily=daily.tail(history_days),
            spikes=episodes(recent_scores) if not recent_scores.empty else [],
            spike_scores=recent_scores,
            forecast=forecast,
            attribution=attribution,
            stations=stations,
            warnings=warnings,
        )

    def _weather(
        self, location: Location, past_days: int, warnings: list[str]
    ) -> pd.DataFrame | None:
        if self.weather is None or self.forecaster is None:
            return None
        try:
            return self.weather.weather(
                location, past_days=past_days, forecast_days=FORECAST_HOURS // 24 + 1
            )
        except ProviderError as exc:
            log.warning("weather unavailable", extra={"error": str(exc)})
            warnings.append(
                "The weather forecast is unavailable, so the model is running without "
                "weather inputs. Expect less accurate forecasts."
            )
            return None

    def _forecast(
        self,
        history: pd.DataFrame,
        provider_fc: pd.DataFrame,
        weather: pd.DataFrame | None,
        location: Location,
        standard: AqiStandard,
    ) -> ForecastView:
        start = history.index.max() if len(history) else None
        if start is None:
            index = pd.DatetimeIndex([], tz="UTC")
        else:
            index = pd.date_range(
                start + pd.Timedelta(hours=1), periods=FORECAST_HOURS, freq="h", tz="UTC"
            )
        frame = pd.DataFrame(index=index)
        frame["provider"] = (
            provider_fc["pm2_5"].reindex(index) if "pm2_5" in provider_fc else np.nan
        )
        for col in ("model", "model_lo", "model_hi"):
            frame[col] = np.nan
        message: str | None = None
        origin: pd.Timestamp | None = None
        skill: dict[str, float] = {}
        coverage: float | None = None
        if self.forecaster is None:
            message = (
                "Our forecasting model isn't available (see the app log), so only the "
                "provider's forecast is shown."
            )
        else:
            summary: dict[str, Any] = self.forecaster.metadata.get("summary", {})
            cv = summary.get("cv", {})
            skill = {k: float(v["skill_vs_persistence"]) for k, v in cv.get("buckets", {}).items()}
            coverage = cv.get("interval_coverage_80")
            try:
                fc = self.forecaster.forecast(history, weather, location.timezone)
            except ForecastUnavailableError as exc:
                message = f"The model can't forecast right now: {exc}."
            else:
                origin = fc.origin
                aligned = fc.frame.reindex(index)
                frame["model"] = aligned["pm2_5"]
                frame["model_lo"] = aligned["lo"]
                frame["model_hi"] = aligned["hi"]
        pm_hist = history["pm2_5"] if "pm2_5" in history else pd.Series(dtype=float)
        frame["model_aqi"] = _pm25_aqi_path(pm_hist, frame["model"], standard)
        frame["provider_aqi"] = _pm25_aqi_path(pm_hist, frame["provider"], standard)
        frame["model_category"] = [
            categorize(v, standard).label if pd.notna(v) else "" for v in frame["model_aqi"]
        ]
        frame["provider_category"] = [
            categorize(v, standard).label if pd.notna(v) else "" for v in frame["provider_aqi"]
        ]
        return ForecastView(
            frame=frame,
            model_available=bool(frame["model"].notna().any()),
            message=message,
            origin=origin,
            skill=skill,
            interval_coverage=coverage,
            peak=_peak(frame, standard),
        )

    def compare(
        self, locations: list[Location], standard: AqiStandard, days: int = 14
    ) -> CityComparison:
        if len(locations) > MAX_COMPARE:
            raise ValueError(f"compare at most {MAX_COMPARE} cities")
        rows: list[dict[str, Any]] = []
        daily_cols: dict[str, pd.Series] = {}
        errors: dict[str, str] = {}
        for loc in locations:
            try:
                data = self.provider.air_quality(loc, past_days=days + 1, forecast_days=1)
            except ProviderError as exc:
                errors[loc.label] = str(exc)
                continue
            history = data.history.dropna(how="all")
            current, _ = _current(history, standard)
            pm = history["pm2_5"].dropna() if "pm2_5" in history else pd.Series(dtype=float)
            rows.append(
                {
                    "city": loc.label,
                    "aqi": current.aqi if current else np.nan,
                    "category": current.category.label if current else "Insufficient data",
                    "level": current.category.level if current else 0,
                    "color": current.category.color if current else "#BBBBBB",
                    "dominant": current.dominant.label if current else "",
                    "pm2_5": float(pm.iloc[-1]) if len(pm) else np.nan,
                }
            )
            daily = daily_aqi(history, standard, tz=loc.timezone)
            daily_cols[loc.label] = daily["aqi"].tail(days)
        summary = pd.DataFrame(rows)
        if not summary.empty:
            summary = summary.sort_values("aqi", ascending=False, na_position="last")
        return CityComparison(summary, pd.DataFrame(daily_cols), errors)


def _station(provider: Provider, location: Location, warnings: list[str]) -> StationReading | None:
    try:
        data = provider.air_quality(location, past_days=2, forecast_days=0)
    except ProviderError as exc:
        log.warning("station provider failed", extra={"source": provider.name, "error": str(exc)})
        warnings.append(f"Ground-station data from {provider.name} is unavailable right now.")
        return None
    hist = data.history.dropna(how="all")
    if hist.empty:
        return None
    last = hist.ffill().iloc[-1]

    def value(col: str) -> float | None:
        v = last.get(col)
        return None if v is None or pd.isna(v) else float(v)

    return StationReading(
        provider.name, pd.Timestamp(hist.index[-1]), value("pm2_5"), value("pm10")
    )


def _current(
    history: pd.DataFrame, standard: AqiStandard
) -> tuple[CurrentConditions | None, str | None]:
    """AQI for the most recent hour whose averages meet the standard's data rules."""
    if history.empty:
        return None, "No recent air quality data is available for this location."
    inputs = averaged_inputs(history, standard)
    for ts in reversed(inputs.index[-6:]):  # tolerate a few missing latest hours
        row = inputs.loc[ts]
        values = {k: (None if pd.isna(v) or v < 0 else float(v)) for k, v in row.items()}
        try:
            result = compute_aqi(Concentrations(**values), standard)
        except InsufficientDataError:
            continue
        latest = history.loc[:ts].ffill().iloc[-1]
        readings = []
        for p in Pollutant:
            raw = latest.get(p.value)
            conc = None if raw is None or pd.isna(raw) else float(raw)
            unit = "µg/m³"
            if p is Pollutant.CO and conc is not None:
                conc, unit = conc / 1000, "mg/m³"
            sub = result.sub_indices.get(p)
            readings.append(
                PollutantReading(
                    p, conc, unit, sub.value if sub else None, AVERAGING_LABEL[standard][p]
                )
            )
        current = CurrentConditions(
            time=pd.Timestamp(ts),
            aqi=result.value,
            category=result.category,
            dominant=result.dominant,
            above_scale=result.above_scale,
            advice=advice_for(result.category, standard),
            pollutants=readings,
        )
        return current, None
    if standard == "in_naqi":
        msg = (
            "Not enough recent data for an NAQI value: it needs at least 16 hours of data "
            "for three pollutants, including PM2.5 or PM10."
        )
    else:
        msg = "Not enough recent data to calculate the AQI."
    return None, msg


def _pm25_aqi_path(history: pd.Series, future: pd.Series, standard: AqiStandard) -> pd.Series:
    """PM2.5-based AQI along a forecast path, averaged the way the standard requires."""
    if future.isna().all():
        return pd.Series(np.nan, index=future.index)
    path = pd.concat([history.dropna(), future.dropna()]).sort_index()
    path = path[~path.index.duplicated(keep="last")]
    path = path.asfreq("h")
    epa = standard == "us_epa"
    avg = nowcast(path) if epa else rolling_mean(path, 24, MIN_HOURS_24H_NAQI)
    values = [pm25_sub_index(v, standard) if pd.notna(v) and v >= 0 else np.nan for v in avg]
    return pd.Series(values, index=avg.index, dtype=float).reindex(future.index)


def _peak(
    frame: pd.DataFrame, standard: AqiStandard
) -> tuple[pd.Timestamp, float, Category] | None:
    col = "model_aqi" if frame["model_aqi"].notna().any() else "provider_aqi"
    series = frame[col].dropna()
    if series.empty:
        return None
    pos = int(np.argmax(series.to_numpy()))
    ts = pd.Timestamp(series.index[pos])
    value = float(series.iloc[pos])
    return ts, value, categorize(value, standard)
