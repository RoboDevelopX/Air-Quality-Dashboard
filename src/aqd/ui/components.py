"""Streamlit rendering helpers. Display only: every value arrives computed from core/."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from aqd.core.categories import CATEGORIES, AqiStandard
from aqd.core.health import DISCLAIMER, SENSITIVE_GROUPS
from aqd.core.service import CurrentConditions, Dashboard, ForecastView

BUCKET_TEXT = {"1-6h": "1 to 6 hours", "7-24h": "7 to 24 hours", "25-48h": "25 to 48 hours"}

CSS = """
<style>
.aqd-card {display:flex; flex-wrap:wrap; gap:1rem; align-items:center; padding:1rem 1.25rem;
  border-radius:12px; border:1px solid rgba(128,128,128,.35);}
.aqd-badge {min-width:7.5rem; padding:.75rem 1rem; border-radius:10px; text-align:center;}
.aqd-badge .num {font-size:2.6rem; font-weight:700; line-height:1;}
.aqd-badge .lvl {font-size:.8rem; margin-top:.25rem;}
.aqd-meta {flex:1 1 14rem;}
.aqd-meta .cat {font-size:1.4rem; font-weight:650;}
.aqd-scale {display:flex; gap:3px; margin-top:.5rem;}
.aqd-scale span {flex:1; height:10px; border-radius:3px; opacity:.35;}
.aqd-scale span.on {opacity:1; outline:2px solid currentColor; outline-offset:1px;}
.aqd-small {font-size:.85rem; opacity:.85;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def local_time(ts: pd.Timestamp, tz: str) -> str:
    return pd.Timestamp(ts).tz_convert(tz).strftime("%a %d %b, %H:%M")


def headline(d: Dashboard) -> None:
    c = d.current
    if c is None:
        st.info(d.current_message or "No current reading is available.")
        return
    cat = c.category
    value = f"{c.aqi}+" if c.above_scale else str(c.aqi)
    scale = "".join(
        f'<span class="{"on" if k.key == cat.key else ""}" style="background:{k.color}" '
        f'title="{html.escape(k.label)}"></span>'
        for k in CATEGORIES[d.standard]
    )
    st.markdown(
        f"""
<div class="aqd-card" role="group" aria-label="Current air quality">
  <div class="aqd-badge" style="background:{cat.color}; color:{cat.text_color}">
    <div class="num">{value}</div>
    <div class="lvl">level {cat.level} of 6</div>
  </div>
  <div class="aqd-meta">
    <div class="cat">{html.escape(cat.label)}</div>
    <div class="aqd-small">{html.escape(d.standard_name)} · main pollutant
      {html.escape(c.dominant.label)} · {html.escape(local_time(c.time, d.location.timezone))}
      local time</div>
    <div class="aqd-scale" aria-hidden="true">{scale}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def health(c: CurrentConditions) -> None:
    st.subheader("Health guidance")
    st.markdown(f"**{c.advice.summary}**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sensitive groups**")
        st.write(c.advice.sensitive_groups)
        st.caption(SENSITIVE_GROUPS)
    with col2:
        st.markdown("**Everyone else**")
        st.write(c.advice.everyone)
    st.caption(DISCLAIMER)


def pollutant_table(c: CurrentConditions) -> None:
    st.subheader("Pollutants")
    rows = [
        {
            "Pollutant": r.pollutant.label,
            "Latest hour": None if r.concentration is None else round(r.concentration, 2),
            "Unit": r.unit,
            "Sub-index": r.sub_index,
            "Averaging used for the index": r.averaging,
            "Main": "◆ main" if r.pollutant is c.dominant else "",
        }
        for r in c.pollutants
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def spike_callout(d: Dashboard) -> None:
    if not d.spikes:
        return
    latest = d.spikes[-1]
    when = local_time(latest.peak_time, d.location.timezone)
    extra = f" ({len(d.spikes)} episodes in this period)" if len(d.spikes) > 1 else ""
    st.warning(
        f"**Unusual PM2.5 spike detected:** {latest.peak_value:.0f} µg/m³ at {when}, against a "
        f"typical {latest.baseline:.0f} µg/m³ for the preceding day{extra}. Spikes are often "
        "short episodes (fires, dust, fireworks); check local news or official monitors.",
        icon="⚠️",
    )


def forecast_summary(f: ForecastView, standard: AqiStandard, tz: str) -> None:
    if f.message:
        st.info(f.message)
    if f.peak is not None:
        ts, value, cat = f.peak
        source = "our model" if f.model_available else "the CAMS forecast"
        st.markdown(
            f"**Highest expected in the next 48 hours:** PM2.5-based AQI {value:.0f} "
            f"(**{cat.label}**) around {local_time(ts, tz)}, according to {source}."
        )
    if f.model_available and f.skill:
        parts = []
        for bucket in ("1-6h", "7-24h", "25-48h"):
            if bucket in f.skill:
                s = f.skill[bucket]
                parts.append(
                    f"{BUCKET_TEXT[bucket]} ahead: {s:.0%} lower error than assuming no change"
                    if s > 0
                    else f"{BUCKET_TEXT[bucket]} ahead: no better than assuming no change"
                )
        coverage = (
            f" The shaded 80% range contained the actual value {f.interval_coverage:.0%} "
            "of the time."
            if f.interval_coverage is not None
            else ""
        )
        st.caption("How much to trust this: on held-out data, " + "; ".join(parts) + "." + coverage)
    st.caption(
        "Forecast AQI uses PM2.5 only (the model forecasts PM2.5), so it can understate the "
        "AQI when ozone or another pollutant dominates. 'Recent' values are CAMS model "
        "estimates, not station measurements."
    )


def forecast_table(f: ForecastView, tz: str) -> pd.DataFrame:
    """Every 6th hour of the forecast as a readable table (the chart's text alternative)."""
    frame = f.frame.iloc[5::6]
    return pd.DataFrame(
        {
            "Local time": [local_time(t, tz) for t in frame.index],
            "Model PM2.5 (µg/m³)": frame["model"].round(1),
            "80% range": [
                "" if pd.isna(lo) else f"{lo:.0f}-{hi:.0f}"
                for lo, hi in zip(frame["model_lo"], frame["model_hi"], strict=True)
            ],
            "Model category": frame["model_category"],
            "CAMS PM2.5 (µg/m³)": frame["provider"].round(1),
            "CAMS category": frame["provider_category"],
        }
    )
