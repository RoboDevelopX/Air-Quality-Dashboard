"""Averaging rules that turn hourly concentrations into the inputs each AQI standard needs.

All functions take hourly series on a UTC DatetimeIndex. Gaps are made explicit first
(`to_hourly_grid`) so that "8-hour average" always means the last 8 clock hours, not the
last 8 rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum valid hours per window. EPA uses 75% completeness (18 of 24, 6 of 8);
# CPCB requires at least 16 hours of data for a 24-hour NAQI sub-index.
MIN_HOURS_24H_EPA = 18
MIN_HOURS_24H_NAQI = 16
MIN_HOURS_8H = 6

NOWCAST_HOURS = 12
NOWCAST_MIN_WEIGHT_PM = 0.5


def to_hourly_grid(obj: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto a complete hourly grid so missing hours become NaN rows."""
    if obj.empty:
        return obj
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex")
    idx = obj.index.floor("h")
    obj = obj.groupby(idx).mean()
    full = pd.date_range(idx.min(), idx.max(), freq="h", tz=idx.tz)
    return obj.reindex(full)


def rolling_mean(series: pd.Series, hours: int, min_valid: int) -> pd.Series:
    """Trailing mean over the last `hours` hours (inclusive of the current hour)."""
    return series.rolling(window=hours, min_periods=min_valid).mean()


def nowcast(series: pd.Series, min_weight: float = NOWCAST_MIN_WEIGHT_PM) -> pd.Series:
    """EPA NowCast for particulate matter on an hourly series.

    For each hour, with c_0 the current hour and c_11 eleven hours earlier:
      w = min(c) / max(c) over valid hours, floored at `min_weight` (0.5 for PM);
      NowCast = sum(w^i * c_i) / sum(w^i) over valid hours.
    At least 2 of the 3 most recent hours must be valid, otherwise the result is NaN.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    padded = np.concatenate([np.full(NOWCAST_HOURS - 1, np.nan), values])
    # windows[t] = values[t-11 .. t]; reverse so column 0 is the current hour.
    windows = np.lib.stride_tricks.sliding_window_view(padded, NOWCAST_HOURS)[:, ::-1]
    for t in range(n):
        w = windows[t]
        if np.count_nonzero(~np.isnan(w[:3])) < 2:
            continue
        valid = ~np.isnan(w)
        c = w[valid]
        c_max = c.max()
        if c_max <= 0:
            out[t] = 0.0
            continue
        weight = max(c.min() / c_max, min_weight)
        powers = weight ** np.arange(NOWCAST_HOURS)[valid]
        out[t] = float(np.sum(powers * c) / np.sum(powers))
    return pd.Series(out, index=series.index, name=series.name)
