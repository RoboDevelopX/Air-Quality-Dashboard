"""Spike detection on recent hourly readings.

Two causal checks (each hour is judged only against hours before it):
- level: robust z-score of the reading against the trailing 24-hour median, scaled by the
  median absolute deviation (MAD) of that residual over the trailing 7 days;
- jump: the hour-over-hour rise is far outside the usual hour-over-hour changes.

A minimum absolute excess stops tiny wiggles at clean-air levels being called spikes.
CAMS global data is 3-hourly and interpolated to hourly, so genuine one-hour spikes are
rare in it; most flags will be multi-hour episodes (fires, dust, festivals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826  # MAD * 1.4826 estimates the standard deviation for normal data


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    baseline_hours: int = 24
    scale_hours: int = 7 * 24
    z_threshold: float = 3.5
    jump_threshold: float = 5.0
    min_excess: float = 10.0  # µg/m³ above the baseline before anything counts as a spike
    min_history: int = 48


def detect_spikes(series: pd.Series, config: AnomalyConfig | None = None) -> pd.DataFrame:
    """Score every hour; returns columns value, baseline, z, jump_z, is_spike, reason."""
    cfg = config or AnomalyConfig()
    s = series.astype(float)
    prev = s.shift(1)  # strictly earlier hours, so a spike can't hide itself in its baseline
    baseline = prev.rolling(cfg.baseline_hours, min_periods=cfg.baseline_hours // 2).median()
    resid = s - baseline
    prev_resid = resid.shift(1)
    mad = (prev_resid - prev_resid.rolling(cfg.scale_hours, min_periods=24).median()).abs()
    scale = MAD_TO_SIGMA * mad.rolling(cfg.scale_hours, min_periods=24).median()
    z = resid / scale.where(scale > 0)

    diff = s.diff()
    prev_diff = diff.shift(1)
    diff_center = prev_diff.rolling(cfg.scale_hours, min_periods=24).median()
    diff_mad = (prev_diff - diff_center).abs().rolling(cfg.scale_hours, min_periods=24).median()
    jump_z = (diff - diff_center) / (MAD_TO_SIGMA * diff_mad).where(diff_mad > 0)

    enough = pd.Series(np.arange(len(s)) >= cfg.min_history, index=s.index)
    big_enough = resid >= cfg.min_excess
    level = enough & big_enough & (z >= cfg.z_threshold)
    jump = enough & big_enough & (jump_z >= cfg.jump_threshold)
    reason = np.select([level & jump, level, jump], ["level+jump", "level", "jump"], default="")
    return pd.DataFrame(
        {
            "value": s,
            "baseline": baseline,
            "z": z,
            "jump_z": jump_z,
            "is_spike": level | jump,
            "reason": reason,
        },
        index=s.index,
    )


@dataclass(frozen=True, slots=True)
class SpikeEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    peak_time: pd.Timestamp
    peak_value: float
    baseline: float

    @property
    def hours(self) -> int:
        return int((self.end - self.start) / pd.Timedelta(hours=1)) + 1


def episodes(scored: pd.DataFrame, max_gap_hours: int = 2) -> list[SpikeEpisode]:
    """Merge flagged hours less than `max_gap_hours` apart into episodes, newest last."""
    flagged = scored[scored["is_spike"]]
    if flagged.empty:
        return []
    out: list[SpikeEpisode] = []
    times = pd.DatetimeIndex(flagged.index)
    group_start = 0
    for i in range(1, len(times) + 1):
        if i == len(times) or (times[i] - times[i - 1]) > pd.Timedelta(hours=max_gap_hours):
            chunk = flagged.iloc[group_start:i]
            peak = int(np.argmax(chunk["value"].to_numpy()))
            out.append(
                SpikeEpisode(
                    start=pd.Timestamp(times[group_start]),
                    end=pd.Timestamp(times[i - 1]),
                    peak_time=pd.Timestamp(times[group_start + peak]),
                    peak_value=float(chunk["value"].iloc[peak]),
                    baseline=float(chunk["baseline"].iloc[peak]),
                )
            )
            group_start = i
    return out
