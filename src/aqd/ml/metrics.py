"""Forecast metrics: MAE, RMSE, skill versus a baseline, interval coverage."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

HORIZON_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-6h", 1, 6),
    ("7-24h", 7, 24),
    ("25-48h", 25, 48),
)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def skill(model_mae: float, baseline_mae: float) -> float:
    """1 - MAE_model / MAE_baseline. Positive means better than the baseline."""
    if baseline_mae == 0:
        return float("nan")
    return 1.0 - model_mae / baseline_mae


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def bucket_of(horizon: pd.Series) -> pd.Series:
    labels = pd.Series("", index=horizon.index, dtype=object)
    for name, lo, hi in HORIZON_BUCKETS:
        labels[(horizon >= lo) & (horizon <= hi)] = name
    return labels


def score_table(
    frame: pd.DataFrame,
    predictions: Mapping[str, str],
    *,
    baseline: str = "persistence",
    by: str | None = "bucket",
) -> pd.DataFrame:
    """MAE, RMSE and skill vs `baseline` for each prediction column, optionally grouped.

    `frame` holds `y`, `horizon` and one column per model (names in `predictions`, which
    maps display name -> column). Rows with a missing target or prediction are dropped
    per model, so every model is scored on the rows where all models have a value.
    """
    cols = list(predictions.values())
    data = frame.dropna(subset=["y", *cols]).copy()
    if by == "bucket":
        data["bucket"] = bucket_of(data["horizon"])
    groups = [("all", data)] if by is None else [*data.groupby(by, sort=True), ("all", data)]
    rows = []
    for key, group in groups:
        y = group["y"].to_numpy()
        base_mae = mae(y, group[predictions[baseline]].to_numpy()) if baseline else np.nan
        for name, col in predictions.items():
            yhat = group[col].to_numpy()
            m = mae(y, yhat)
            rows.append(
                {
                    "group": key,
                    "model": name,
                    "n": len(group),
                    "mae": m,
                    "rmse": rmse(y, yhat),
                    f"skill_vs_{baseline}": skill(m, base_mae),
                }
            )
    return pd.DataFrame(rows)
