"""Rolling-origin (expanding window) cross-validation for direct multi-horizon forecasts.

A training row with origin t uses PM2.5 at t+h as its target. If t+h falls inside the test
window, the model has seen test-period values, so training origins must end at least
`gap_hours` (>= the maximum horizon) before the test window starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Fold:
    number: int
    train_end: pd.Timestamp  # last allowed training origin (inclusive)
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def make_folds(
    origins: pd.Series | pd.DatetimeIndex,
    *,
    n_folds: int = 6,
    test_days: int = 30,
    gap_hours: int = 48,
    min_train_days: int = 180,
) -> list[Fold]:
    """Folds whose test windows tile the most recent `n_folds * test_days` days."""
    times = pd.DatetimeIndex(origins)
    if times.empty:
        raise ValueError("no origins to split")
    end = times.max().floor("h") + pd.Timedelta(hours=1)
    first_test_start = end - pd.Timedelta(days=test_days * n_folds)
    if first_test_start - pd.Timedelta(hours=gap_hours) < times.min() + pd.Timedelta(
        days=min_train_days
    ):
        raise ValueError(
            f"not enough history for {n_folds} folds of {test_days} days with "
            f"{min_train_days} training days"
        )
    folds = []
    for i in range(n_folds):
        test_start = first_test_start + pd.Timedelta(days=test_days * i)
        test_end = test_start + pd.Timedelta(days=test_days)
        train_end = test_start - pd.Timedelta(hours=gap_hours + 1)
        folds.append(Fold(i + 1, train_end, test_start, test_end))
    return folds


def split(frame: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, test) over rows of a stacked feature frame.

    Train: origin <= train_end AND target time < test_start (the second condition is
    implied by the gap but is checked explicitly so a misconfigured gap can't leak).
    Test: origin inside the test window.
    """
    origin = pd.DatetimeIndex(frame["origin"])
    valid = pd.DatetimeIndex(frame["valid_time"])
    train = (origin <= fold.train_end) & (valid < fold.test_start)
    test = (origin >= fold.test_start) & (origin < fold.test_end)
    return np.asarray(train), np.asarray(test)
