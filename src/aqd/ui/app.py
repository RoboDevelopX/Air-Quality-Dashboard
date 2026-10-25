"""The Streamlit page: controls, tabs and wiring. Rendering only; logic lives in core/."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from aqd.config import Settings, load_dotenv
from aqd.core.categories import STANDARD_NAMES, AqiStandard
from aqd.core.pollutants import Pollutant
from aqd.core.service import MAX_COMPARE, Dashboard, DashboardService
from aqd.log import configure_logging
from aqd.ml.inference import load_forecaster
from aqd.providers.base import Location, Provider
from aqd.providers.http import HttpClient, ProviderError
from aqd.providers.open_meteo import OpenMeteoAirQuality, OpenMeteoGeocoder, OpenMeteoWeather
from aqd.providers.openaq import OpenAQProvider
from aqd.storage.cache import SqliteCache
from aqd.ui import charts, components

log = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def build_service(settings: Settings) -> DashboardService:
    """One service (HTTP client, cache, model) per process, shared by all sessions."""
    http = HttpClient(SqliteCache(settings.cache_path), timeout_s=settings.http_timeout_s)
    forecaster = load_forecaster(settings.model_dir)
    extra: list[Provider] = []
    if settings.openaq_api_key:
        extra.append(OpenAQProvider(http, settings.openaq_api_key))
    return DashboardService(
        OpenMeteoAirQuality(
            http,
            ttl_forecast_s=settings.cache_ttl_forecast_s,
            ttl_history_s=settings.cache_ttl_history_s,
        ),
        OpenMeteoGeocoder(http, ttl_s=settings.cache_ttl_geocode_s),
        weather=OpenMeteoWeather(http, ttl_forecast_s=settings.cache_ttl_forecast_s),
        forecaster=forecaster,
        extra_providers=extra,
    )


@st.cache_data(ttl=600, show_spinner=False)
def load_dashboard(
    _service: DashboardService, location: Location, standard: AqiStandard, days: int
) -> Dashboard:
    return _service.dashboard(location, standard, history_days=days)


@st.cache_data(ttl=86400, show_spinner=False)
def search(_service: DashboardService, query: str) -> list[Location]:
    return _service.search(query, count=8)


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings.log_level, settings.log_format)
    st.set_page_config(page_title="Air Quality Dashboard", page_icon="🌫️", layout="wide")
    components.inject_css()
    service = build_service(settings)

    st.title("Air Quality Dashboard")
    st.caption(
        "Current, historical and forecast air quality for any city, with a PM2.5 forecasting "
        "model. Data: CAMS via Open-Meteo."
    )

    location, standard, days = controls(service, settings)
    if location is None:
        return

    with st.spinner(f"Loading air quality for {location.label}…"):
        try:
            dash = load_dashboard(service, location, standard, days)
        except ProviderError as exc:
            log.error("dashboard failed", extra={"city": location.label, "error": str(exc)})
            st.error(
                f"Couldn't load air quality data for {location.label}. The data provider "
                "may be busy or unreachable. Please try again in a minute."
            )
            if st.button("Try again"):
                load_dashboard.clear()
                st.rerun()
            return

    for warning in dash.warnings:
        st.warning(warning, icon="⚠️")

    now_tab, history_tab, forecast_tab, compare_tab, about_tab = st.tabs(
        ["Now", "History", "Forecast", "Compare cities", "About"]
    )
    with now_tab:
        render_now(dash)
    with history_tab:
        render_history(dash, days)
    with forecast_tab:
        render_forecast(dash)
    with compare_tab:
        render_compare(service, standard, location)
    with about_tab:
        render_about(dash, service)


def controls(
    service: DashboardService, settings: Settings
) -> tuple[Location | None, AqiStandard, int]:
    col_city, col_std, col_days = st.columns([3, 2, 2])
    with col_city:
        query = st.text_input("City", value=settings.default_city, max_chars=80)
    with col_std:
        standards: list[AqiStandard] = ["us_epa", "in_naqi"]
        standard = st.selectbox(
            "AQI standard",
            standards,
            index=standards.index(settings.aqi_standard),
            format_func=STANDARD_NAMES.__getitem__,
        )
    with col_days:
        days = st.radio("History", [7, 30], index=1, horizontal=True, format_func="{} days".format)

    try:
        matches = search(service, query) if query.strip() else []
    except ProviderError:
        st.error("City search is unavailable right now. Please try again shortly.")
        return None, standard, days
    if not matches:
        st.info(
            "Type a city name to begin." if not query.strip() else f"No places match “{query}”."
        )
        return None, standard, days
    location = matches[0]
    if len(matches) > 1:
        location = st.selectbox(
            "Did you mean", matches, format_func=lambda p: p.label, key=f"pick-{query}"
        )
    return location, standard, days


def render_now(d: Dashboard) -> None:
    components.headline(d)
    components.spike_callout(d)
    if d.current is not None:
        components.health(d.current)
        components.pollutant_table(d.current)
    for s in d.stations:
        parts = []
        if s.pm2_5 is not None:
            parts.append(f"PM2.5 {s.pm2_5:.0f} µg/m³")
        if s.pm10 is not None:
            parts.append(f"PM10 {s.pm10:.0f} µg/m³")
        st.caption(
            f"Nearest ground station ({s.source}): {', '.join(parts)} at "
            f"{components.local_time(s.time, d.location.timezone)}."
        )


def render_history(d: Dashboard, days: int) -> None:
    if d.hourly.empty or d.hourly["aqi"].isna().all():
        st.info("Not enough history to chart for this location.")
        return
    tz = d.location.timezone
    st.plotly_chart(
        charts.daily_aqi_bars(d.daily, d.standard, f"Daily {d.standard_name}, last {days} days"),
        width="stretch",
        theme="streamlit",
    )
    st.plotly_chart(
        charts.hourly_aqi_line(d.hourly, d.spike_scores, d.standard, tz, "Hourly AQI"),
        width="stretch",
        theme="streamlit",
    )
    options = [p.value for p in Pollutant if p.value in d.history]
    chosen = st.multiselect(
        "Pollutants",
        options,
        default=[o for o in ("pm2_5", "pm10") if o in options],
        format_func=lambda v: Pollutant(v).label,
    )
    if chosen:
        st.plotly_chart(
            charts.pollutant_lines(d.history, chosen, tz),
            width="stretch",
            theme="streamlit",
        )
    with st.expander("Daily values as a table"):
        table = d.daily[["aqi", "category", "dominant"]].copy()
        table.columns = ["AQI", "Category", "Main pollutant"]
        st.dataframe(table, width="stretch")


def render_forecast(d: Dashboard) -> None:
    f = d.forecast
    if f.frame.empty:
        st.info("No forecast is available for this location.")
        return
    tz = d.location.timezone
    components.forecast_summary(f, d.standard, tz)
    observed = d.history["pm2_5"].iloc[-48:] if "pm2_5" in d.history else pd.Series(dtype=float)
    st.plotly_chart(
        charts.forecast_chart(observed, f.frame, tz, show_model=f.model_available),
        width="stretch",
        theme="streamlit",
    )
    with st.expander("Forecast as a table (every 6 hours)"):
        st.dataframe(components.forecast_table(f, tz), hide_index=True, width="stretch")


def render_compare(service: DashboardService, standard: AqiStandard, current: Location) -> None:
    key = "compare_cities"
    if key not in st.session_state:
        st.session_state[key] = [current]
    cities: list[Location] = st.session_state[key]

    col_add, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_add:
        query = st.text_input("Add a city to compare", key="compare_query", max_chars=80)
    with col_btn:
        add = st.button("Add", disabled=len(cities) >= MAX_COMPARE, width="stretch")
    if add and query.strip():
        try:
            found = search(service, query)
        except ProviderError:
            found = []
            st.error("City search is unavailable right now.")
        if not found:
            st.warning(f"No places match “{query}”.")
        elif found[0].key not in {c.key for c in cities}:
            cities.append(found[0])
    if len(cities) >= MAX_COMPARE:
        st.caption(f"Up to {MAX_COMPARE} cities can be compared at once.")

    remove = st.multiselect(
        "Remove", cities, format_func=lambda p: p.label, placeholder="Choose cities to remove"
    )
    if remove:
        st.session_state[key] = cities = [c for c in cities if c not in remove]
    if not cities:
        st.info("Add a city to start comparing.")
        return
    with st.spinner("Loading cities…"):
        result = service.compare(cities, standard, days=14)
    for label, err in result.errors.items():
        st.warning(f"{label}: data unavailable ({err}).")
    if result.summary.empty:
        return
    st.plotly_chart(charts.compare_current_bars(result.summary), width="stretch", theme="streamlit")
    if not result.daily.empty:
        st.plotly_chart(
            charts.compare_daily_lines(result.daily, standard),
            width="stretch",
            theme="streamlit",
        )
    table = result.summary[["city", "aqi", "category", "dominant", "pm2_5"]].rename(
        columns={
            "city": "City",
            "aqi": "AQI",
            "category": "Category",
            "dominant": "Main pollutant",
            "pm2_5": "PM2.5 (µg/m³)",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")


def render_about(d: Dashboard, service: DashboardService) -> None:
    st.subheader("About the forecast model")
    meta: dict[str, Any] = (
        service.forecaster.metadata.get("summary", {}) if service.forecaster else {}
    )
    if meta:
        cv = meta.get("cv", {})
        st.markdown(
            f"A LightGBM model trained on hourly CAMS data for {len(meta.get('cities', []))} "
            f"cities ({meta.get('data_start', '?')[:10]} to {meta.get('data_end', '?')[:10]}), "
            f"evaluated with {cv.get('n_folds', '?')}-fold rolling-origin cross-validation. "
            "See MODEL_CARD.md in the repository for data, features, metrics and limitations."
        )
        rows = [
            {
                "Horizon": k,
                "MAE (µg/m³)": v["mae"],
                "Skill vs persistence": f"{v['skill_vs_persistence']:.0%}",
            }
            for k, v in cv.get("buckets", {}).items()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No trained model is loaded; the app shows the provider's forecast only.")
    st.subheader("Data sources")
    for line in d.attribution:
        st.markdown(f"- {line}")
    st.caption(
        f"Data fetched {d.issued_at:%Y-%m-%d %H:%M} UTC. AQI calculations follow US EPA "
        "(40 CFR 58 App. G, 2024) and CPCB NAQI breakpoint tables."
    )
