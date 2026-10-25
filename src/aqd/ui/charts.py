"""Plotly figure builders. Pure functions: data in, figure out, no Streamlit.

Conventions: one y-axis per chart, thin 2px lines, legends for 2+ series, hover on every
mark, and AQI categories always named in text (bands are labelled, hovers say the
category), so colour is never the only carrier of meaning.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from aqd.core.categories import CATEGORIES, AqiStandard, categorize

# Categorical series colours (validated colour-blind-safe order), plus neutral ink.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MODEL_COLOR = SERIES[0]
PROVIDER_COLOR = SERIES[1]
OBSERVED_COLOR = "#6b6a66"

_LAYOUT = {
    "margin": {"l": 8, "r": 8, "t": 76, "b": 8},
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
    "height": 360,
}
_TITLE = {"y": 0.97, "yanchor": "top", "x": 0, "xanchor": "left", "pad": {"l": 4}}
# Bands thinner than this share of the y-range are too short to carry a readable label.
_MIN_LABELLED_BAND = 0.07


def _local(index: pd.Index, tz: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).tz_convert(tz).tz_localize(None)


def _category_bands(fig: go.Figure, standard: AqiStandard, y_max: float) -> None:
    """Faint horizontal bands per AQI category, each labelled with its name."""
    for cat in CATEGORIES[standard]:
        if cat.aqi_lo > y_max:
            break
        top = min(cat.aqi_hi, y_max)
        labelled = (top - cat.aqi_lo) / y_max >= _MIN_LABELLED_BAND
        fig.add_hrect(
            y0=cat.aqi_lo,
            y1=top,
            fillcolor=cat.color,
            opacity=0.10,
            line_width=0,
            layer="below",
            annotation_text=cat.label if labelled else None,
            annotation_position="right",
            annotation_xanchor="right",
            annotation_font_size=10,
        )


def daily_aqi_bars(daily: pd.DataFrame, standard: AqiStandard, title: str) -> go.Figure:
    data = daily.dropna(subset=["aqi"])
    cats = [categorize(v, standard) for v in data["aqi"]]
    fig = go.Figure(
        go.Bar(
            x=pd.to_datetime(pd.Index(data.index).astype(str)),
            y=data["aqi"],
            marker={"color": [c.color for c in cats], "line": {"color": "#555555", "width": 0.5}},
            customdata=[[c.label, d] for c, d in zip(cats, data["dominant"], strict=True)],
            hovertemplate="%{x|%a %d %b}: AQI %{y:.0f}<br>%{customdata[0]}"
            "<br>Main pollutant: %{customdata[1]}<extra></extra>",
            name="Daily AQI",
        )
    )
    y_max = max(float(data["aqi"].max()) * 1.15, 60) if len(data) else 100
    _category_bands(fig, standard, y_max)
    fig.update_layout(**_LAYOUT, title={"text": title, **_TITLE}, showlegend=False, bargap=0.25)
    fig.update_yaxes(title="AQI", range=[0, y_max], rangemode="tozero")
    return fig


def hourly_aqi_line(
    hourly: pd.DataFrame,
    spikes: pd.DataFrame,
    standard: AqiStandard,
    tz: str,
    title: str,
) -> go.Figure:
    data = hourly.dropna(subset=["aqi"])
    x = _local(data.index, tz)
    labels = [categorize(v, standard).label for v in data["aqi"]]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=data["aqi"],
            mode="lines",
            line={"color": MODEL_COLOR, "width": 2},
            customdata=labels,
            hovertemplate="AQI %{y:.0f} (%{customdata})<extra></extra>",
            name="Hourly AQI",
        )
    )
    flagged = spikes[spikes.get("is_spike", pd.Series(dtype=bool))] if len(spikes) else spikes
    if len(flagged):
        at = data["aqi"].reindex(flagged.index)
        fig.add_trace(
            go.Scatter(
                x=_local(flagged.index, tz),
                y=at,
                mode="markers",
                marker={"symbol": "x", "size": 10, "color": "#c0392b", "line": {"width": 2}},
                customdata=flagged["value"],
                hovertemplate="Spike: PM2.5 %{customdata:.0f} µg/m³<extra></extra>",
                name="PM2.5 spike",
            )
        )
    y_max = max(float(data["aqi"].max()) * 1.15, 60) if len(data) else 100
    _category_bands(fig, standard, y_max)
    fig.update_layout(**_LAYOUT, title={"text": title, **_TITLE})
    fig.update_yaxes(title="AQI", range=[0, y_max])
    return fig


def pollutant_lines(history: pd.DataFrame, columns: Sequence[str], tz: str) -> go.Figure:
    from aqd.core.pollutants import Pollutant

    fig = go.Figure()
    for i, col in enumerate(columns):
        if col not in history:
            continue
        label = Pollutant(col).label
        fig.add_trace(
            go.Scatter(
                x=_local(history.index, tz),
                y=history[col],
                mode="lines",
                line={"color": SERIES[i % len(SERIES)], "width": 2},
                name=label,
                hovertemplate=f"{label}: %{{y:.1f}} µg/m³<extra></extra>",
            )
        )
    fig.update_layout(**_LAYOUT, title={"text": "Hourly concentrations", **_TITLE})
    fig.update_yaxes(title="µg/m³", rangemode="tozero")
    return fig


def forecast_chart(
    observed: pd.Series, forecast: pd.DataFrame, tz: str, *, show_model: bool
) -> go.Figure:
    """Recent PM2.5, the model forecast with its 80% interval, and the provider forecast."""
    fig = go.Figure()
    obs = observed.dropna()
    fig.add_trace(
        go.Scatter(
            x=_local(obs.index, tz),
            y=obs,
            mode="lines",
            line={"color": OBSERVED_COLOR, "width": 2},
            name="Recent (CAMS)",
            hovertemplate="Recent: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    x = _local(forecast.index, tz)
    if show_model:
        fig.add_trace(
            go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(forecast["model_hi"]) + list(forecast["model_lo"][::-1]),
                fill="toself",
                fillcolor="rgba(42,120,214,0.15)",
                line={"width": 0},
                hoverinfo="skip",
                name="80% interval",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=forecast["model"],
                mode="lines",
                line={"color": MODEL_COLOR, "width": 2},
                name="Our model",
                customdata=forecast[["model_lo", "model_hi"]],
                hovertemplate="Model: %{y:.1f} µg/m³ (80%: %{customdata[0]:.0f}"
                "-%{customdata[1]:.0f})<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast["provider"],
            mode="lines",
            line={"color": PROVIDER_COLOR, "width": 2, "dash": "dash"},
            name="CAMS forecast",
            hovertemplate="CAMS: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    if len(forecast):
        fig.add_vline(x=x[0], line={"color": "#999999", "width": 1, "dash": "dot"})
    fig.update_layout(**_LAYOUT, title={"text": "PM2.5: recent and next 48 hours", **_TITLE})
    fig.update_yaxes(title="PM2.5 (µg/m³)", rangemode="tozero")
    return fig


def compare_current_bars(summary: pd.DataFrame) -> go.Figure:
    data = summary.dropna(subset=["aqi"]).sort_values("aqi")
    fig = go.Figure(
        go.Bar(
            x=data["aqi"],
            y=data["city"],
            orientation="h",
            marker={"color": data["color"], "line": {"color": "#555555", "width": 0.5}},
            text=[f"{a:.0f} · {c}" for a, c in zip(data["aqi"], data["category"], strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: AQI %{x:.0f}<extra></extra>",
        )
    )
    layout = {**_LAYOUT, "hovermode": "closest", "height": 80 + 50 * max(len(data), 1)}
    fig.update_layout(**layout, title={"text": "Current AQI", **_TITLE}, showlegend=False)
    fig.update_xaxes(
        title="AQI", range=[0, max(float(data["aqi"].max()) * 1.4, 60) if len(data) else 100]
    )
    return fig


def compare_daily_lines(daily: pd.DataFrame, standard: AqiStandard) -> go.Figure:
    fig = go.Figure()
    for i, city in enumerate(daily.columns):
        series = daily[city].dropna()
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(pd.Index(series.index).astype(str)),
                y=series,
                mode="lines+markers",
                marker={"size": 8},
                line={"color": SERIES[i % len(SERIES)], "width": 2},
                name=str(city),
                hovertemplate=f"{city}: AQI %{{y:.0f}}<extra></extra>",
            )
        )
    y_max = max(float(daily.max().max()) * 1.15, 60) if daily.notna().any().any() else 100
    _category_bands(fig, standard, y_max)
    fig.update_layout(**_LAYOUT, title={"text": "Daily AQI", **_TITLE})
    fig.update_yaxes(title="AQI", range=[0, y_max])
    return fig
