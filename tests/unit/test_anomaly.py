import numpy as np
import pandas as pd

from aqd.core.anomaly import AnomalyConfig, detect_spikes, episodes


def noisy(n: int = 24 * 10, seed: int = 0, level: float = 40.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    return pd.Series(level + 8 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2, n), index=idx)


def test_no_spikes_in_ordinary_noise() -> None:
    assert not detect_spikes(noisy())["is_spike"].any()


def test_injected_spike_is_flagged() -> None:
    s = noisy()
    s.iloc[200:203] += 80
    scored = detect_spikes(s)
    flagged = scored.index[scored["is_spike"]]
    assert s.index[200] in flagged
    assert scored.loc[s.index[200], "reason"] in {"level", "level+jump"}
    eps = episodes(scored)
    assert len(eps) == 1
    assert eps[0].start == s.index[200]
    assert eps[0].hours >= 1
    assert eps[0].peak_value > eps[0].baseline + 60


def test_small_excess_is_ignored_even_if_statistically_unusual() -> None:
    s = pd.Series(5.0, index=pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC"))
    s += np.random.default_rng(0).normal(0, 0.1, len(s))
    s.iloc[200] = 9.0  # huge z-score but only +4 µg/m³
    assert not detect_spikes(s)["is_spike"].any()


def test_needs_history_before_flagging() -> None:
    s = noisy()
    s.iloc[10] += 200
    assert not detect_spikes(s)["is_spike"].iloc[:48].any()


def test_episodes_merge_close_hours_and_split_far_ones() -> None:
    s = noisy(n=24 * 14)
    s.iloc[200] += 90
    s.iloc[202] += 90  # 2 hours later: same episode
    s.iloc[300] += 90  # far later: new episode
    eps = episodes(detect_spikes(s, AnomalyConfig(min_excess=20)))
    assert len(eps) == 2
    assert eps[0].start == s.index[200] and eps[0].end == s.index[202]


def test_episodes_empty() -> None:
    assert episodes(detect_spikes(noisy())) == []
