"""Cross-validated evaluation of baselines, SARIMAX and LightGBM, plus provider comparison."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aqd.ml.cv import Fold, split
from aqd.ml.dataset import CityData
from aqd.ml.metrics import coverage, mae, score_table
from aqd.ml.models import LgbmForecaster, sarimax_forecast

log = logging.getLogger(__name__)

MODEL_COLUMNS = {
    "persistence": "persistence",
    "seasonal naive": "seasonal_naive",
    "SARIMAX": "sarimax",
    "LightGBM": "lgbm",
}


@dataclass
class CvResult:
    predictions: pd.DataFrame  # eval rows with y, baselines, lgbm, lgbm_lo/hi, sarimax, fold
    folds: list[Fold]

    def scores(self, *, by: str | None = "bucket") -> pd.DataFrame:
        """Scores for baselines + LightGBM on all eval rows."""
        cols = {k: v for k, v in MODEL_COLUMNS.items() if v != "sarimax"}
        return score_table(self.predictions, cols, by=by)

    def scores_with_sarimax(self, *, by: str | None = "bucket") -> pd.DataFrame:
        """All four models on the subset of origins where SARIMAX was run."""
        subset = self.predictions[self.predictions["sarimax"].notna()]
        return score_table(subset, MODEL_COLUMNS, by=by)

    def scores_by_city(self) -> pd.DataFrame:
        cols = {k: v for k, v in MODEL_COLUMNS.items() if v != "sarimax"}
        return score_table(self.predictions, cols, by="city")

    def interval_coverage(self) -> float:
        p = self.predictions.dropna(subset=["y", "lgbm_lo", "lgbm_hi"])
        return coverage(p["y"].to_numpy(), p["lgbm_lo"].to_numpy(), p["lgbm_hi"].to_numpy())


def run_cv(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    folds: list[Fold],
    cities: Mapping[str, CityData] | None = None,
    *,
    sarimax_origin_hour: int | None = 0,
    sarimax_every_days: int = 1,
    lgbm_rounds: int = 600,
    lgbm_params: dict[str, object] | None = None,
) -> CvResult:
    """Fit per fold on `train_frame` rows, predict `eval_frame` test rows.

    Intervals for fold k are calibrated on out-of-fold residuals from folds < k only, so
    the reported coverage is itself out-of-sample (fold 1 gets no interval).
    SARIMAX runs once every `sarimax_every_days` days per city (origins at
    `sarimax_origin_hour` UTC) because each origin needs its own fit; None skips it.
    """
    outputs = []
    oof_frames: list[pd.DataFrame] = []
    oof_deltas: list[np.ndarray] = []
    for fold in folds:
        train_mask, _ = split(train_frame, fold)
        _, test_mask = split(eval_frame, fold)
        test = eval_frame[test_mask].copy()
        if test.empty or not train_mask.any():
            log.warning("skipping empty fold", extra={"fold": fold.number})
            continue
        model = LgbmForecaster.fit(
            train_frame[train_mask], num_boost_round=lgbm_rounds, params=lgbm_params
        )
        delta = model.predict_delta(test)
        test["lgbm"] = model.predict(test)
        test["fold"] = fold.number
        if oof_frames:
            model.calibrate(pd.concat(oof_frames), np.concatenate(oof_deltas))
            test["lgbm_lo"], test["lgbm_hi"] = model.predict_interval(test)
        else:
            test["lgbm_lo"] = test["lgbm_hi"] = np.nan
        oof_frames.append(test)
        oof_deltas.append(delta)
        test["sarimax"] = np.nan
        if cities is not None and sarimax_origin_hour is not None:
            _add_sarimax(test, cities, sarimax_origin_hour, sarimax_every_days)
        outputs.append(test)
        log.info(
            "fold done",
            extra={"fold": fold.number, "rows": len(test), "mae_lgbm": _safe_mae(test, "lgbm")},
        )
    if not outputs:
        raise ValueError("no fold produced predictions")
    return CvResult(pd.concat(outputs, ignore_index=True), folds)


def _add_sarimax(
    test: pd.DataFrame, cities: Mapping[str, CityData], hour: int, every_days: int
) -> None:
    origin_times = pd.DatetimeIndex(test["origin"])
    day_number = (origin_times - origin_times.min()).days
    chosen = (origin_times.hour == hour) & (day_number % every_days == 0)
    origins = test.loc[chosen, ["city", "origin"]]
    for city, origin in origins.drop_duplicates().itertuples(index=False):
        series = cities[city].hourly["pm2_5"]
        history = series[series.index <= origin]
        fc = sarimax_forecast(history)
        rows = (test["city"] == city) & (test["origin"] == origin)
        test.loc[rows, "sarimax"] = fc[test.loc[rows, "horizon"].to_numpy() - 1]


def _safe_mae(frame: pd.DataFrame, col: str) -> float:
    p = frame.dropna(subset=["y", col])
    return round(mae(p["y"].to_numpy(), p[col].to_numpy()), 3) if len(p) else float("nan")


def align_provider_rows(
    snapshots: pd.DataFrame, actual: pd.Series, model_predictions: pd.DataFrame
) -> pd.DataFrame:
    """Join provider and model forecasts for the same issue/valid times with the actuals.

    `snapshots`: provider forecasts as issued, columns issued_at, valid_time, pm2_5.
    `model_predictions`: our forecasts made with data up to issued_at, same columns.
    `actual`: PM2.5 later recorded for each valid_time (UTC index).
    Keeps only rows where both forecasts and the actual value exist.
    """
    keys = ["issued_at", "valid_time"]
    merged = snapshots[[*keys, "pm2_5"]].merge(
        model_predictions[[*keys, "pm2_5"]], on=keys, suffixes=("_provider", "_model")
    )
    merged["y"] = actual.reindex(pd.DatetimeIndex(merged["valid_time"])).to_numpy()
    merged = merged.dropna(subset=["y", "pm2_5_provider", "pm2_5_model"])
    lead = pd.to_datetime(merged["valid_time"]) - pd.to_datetime(merged["issued_at"])
    merged["horizon"] = (lead.dt.total_seconds() // 3600).astype(int)
    return merged[merged["horizon"] >= 1].reset_index(drop=True)


def score_provider_rows(rows: pd.DataFrame, *, by: str | None = "bucket") -> pd.DataFrame:
    return score_table(
        rows,
        {"provider (CAMS)": "pm2_5_provider", "LightGBM": "pm2_5_model"},
        baseline="provider (CAMS)",
        by=by,
    )


def compare_with_provider(
    snapshots: pd.DataFrame, actual: pd.Series, model_predictions: pd.DataFrame
) -> pd.DataFrame:
    """MAE of the provider's own forecast vs ours, on the same issue times and horizons."""
    return score_provider_rows(align_provider_rows(snapshots, actual, model_predictions))


def plot_predictions(result: CvResult, city: str, horizon: int, path: Path) -> None:
    """Predicted vs actual for one city at a fixed horizon over the last test fold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    p = result.predictions
    p = p[(p["city"] == city) & (p["horizon"] == horizon)]
    p = p[p["fold"] == p["fold"].max()].sort_values("valid_time")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(p["valid_time"], p["y"], label="actual (CAMS)", color="#222222", lw=1.5)
    ax.plot(p["valid_time"], p["lgbm"], label="LightGBM", color="#1f77b4", lw=1.2)
    ax.plot(p["valid_time"], p["persistence"], label="persistence", color="#999999", lw=1, ls="--")
    if p["lgbm_lo"].notna().any():
        ax.fill_between(
            p["valid_time"], p["lgbm_lo"], p["lgbm_hi"], alpha=0.2, label="80% interval"
        )
    locator = mdates.AutoDateLocator()  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))  # type: ignore[no-untyped-call]
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"{city}: {horizon}h-ahead forecasts on held-out data")
    ax.legend(loc="upper left", ncols=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_by_horizon(result: CvResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = score_table(
        result.predictions,
        {k: v for k, v in MODEL_COLUMNS.items() if v != "sarimax"},
        by="horizon",
    )
    scores = scores[scores["group"] != "all"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for model, g in scores.groupby("model"):
        ax.plot(g["group"].astype(int), g["mae"], label=model, lw=1.5)
    ax.set_xlabel("horizon (hours)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Error by forecast horizon (all held-out folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
