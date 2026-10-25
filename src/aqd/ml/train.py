"""End-to-end training: CV evaluation, reports, final model artifact.

`scripts/train.py` fetches data and calls `train_pipeline`; keeping the pipeline here makes
it testable end to end with a mocked provider.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aqd.ml.cv import make_folds
from aqd.ml.dataset import CityData, stack
from aqd.ml.evaluate import (
    CvResult,
    align_provider_rows,
    plot_error_by_horizon,
    plot_predictions,
    run_cv,
)
from aqd.ml.inference import Forecaster, ForecastUnavailableError
from aqd.ml.models import LgbmForecaster
from aqd.providers.base import POLLUTANT_COLUMNS

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    n_folds: int = 6
    test_days: int = 30
    gap_hours: int = 48
    min_train_days: int = 180
    train_stride_h: int = 4
    horizons_per_origin: int = 12
    eval_stride_h: int = 6
    sarimax_every_days: int = 1
    lgbm_rounds: int = 600
    lgbm_params: Mapping[str, Any] | None = None
    seed: int = 42


@dataclass
class TrainingReport:
    cv: CvResult
    model: LgbmForecaster
    summary: dict[str, Any]


def train_pipeline(
    cities: Mapping[str, CityData],
    *,
    model_dir: Path,
    report_dir: Path,
    config: TrainConfig | None = None,
) -> TrainingReport:
    cfg = config or TrainConfig()
    train_frame = stack(
        cities,
        origin_stride_h=cfg.train_stride_h,
        horizons_per_origin=cfg.horizons_per_origin,
        seed=cfg.seed,
    )
    eval_frame = stack(cities, origin_stride_h=cfg.eval_stride_h)
    log.info("datasets built", extra={"train_rows": len(train_frame), "eval_rows": len(eval_frame)})
    folds = make_folds(
        eval_frame["origin"],
        n_folds=cfg.n_folds,
        test_days=cfg.test_days,
        gap_hours=cfg.gap_hours,
        min_train_days=cfg.min_train_days,
    )
    params = dict(cfg.lgbm_params or {})
    cv = run_cv(
        train_frame,
        eval_frame,
        folds,
        cities,
        sarimax_every_days=cfg.sarimax_every_days,
        lgbm_rounds=cfg.lgbm_rounds,
        lgbm_params=params,
    )

    # Final model: all training rows; intervals from every out-of-fold residual.
    model = LgbmForecaster.fit(train_frame, num_boost_round=cfg.lgbm_rounds, params=params)
    oof = cv.predictions.dropna(subset=["y", "lgbm"])
    oof_delta = np.log1p(oof["lgbm"].to_numpy()) - oof["pm_log_0"].to_numpy()
    model.calibrate(oof, oof_delta)

    summary = _summary(cv, cities, train_frame, cfg)
    model.metadata = {"summary": summary}
    model.save(model_dir)
    write_reports(cv, summary, report_dir, cities)
    return TrainingReport(cv=cv, model=model, summary=summary)


def _summary(
    cv: CvResult, cities: Mapping[str, CityData], train_frame: pd.DataFrame, cfg: TrainConfig
) -> dict[str, Any]:
    scores = cv.scores()
    lgbm = scores[scores["model"] == "LightGBM"].set_index("group")
    starts = [c.hourly.index.min() for c in cities.values()]
    ends = [c.hourly.index.max() for c in cities.values()]
    return {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "lightgbm_version": lgb.__version__,
        "cities": sorted(cities),
        "data_start": str(min(starts)),
        "data_end": str(max(ends)),
        "train_rows": len(train_frame),
        "cv": {
            "n_folds": len(cv.folds),
            "test_days": cfg.test_days,
            "gap_hours": cfg.gap_hours,
            "test_start": str(cv.folds[0].test_start),
            "test_end": str(cv.folds[-1].test_end),
            "interval_coverage_80": round(cv.interval_coverage(), 3),
            "buckets": {
                str(group): {
                    "mae": round(float(row["mae"]), 3),
                    "rmse": round(float(row["rmse"]), 3),
                    "skill_vs_persistence": round(float(row["skill_vs_persistence"]), 3),
                }
                for group, row in lgbm.iterrows()
            },
        },
    }


def write_reports(
    cv: CvResult, summary: Mapping[str, Any], report_dir: Path, cities: Mapping[str, CityData]
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "all_rows": cv.scores(),
        "sarimax_subset": cv.scores_with_sarimax(),
        "by_city": cv.scores_by_city(),
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": summary,
                **{name: t.round(4).to_dict(orient="records") for name, t in tables.items()},
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    lines = ["# Evaluation results", ""]
    lines += [
        f"Rolling-origin CV, {summary['cv']['n_folds']} folds x {summary['cv']['test_days']} "
        f"days, {summary['cv']['gap_hours']}h gap; held-out period "
        f"{summary['cv']['test_start'][:10]} to {summary['cv']['test_end'][:10]}. "
        "Errors are PM2.5 in µg/m³ against CAMS (not station measurements).",
        "",
        f"80% interval coverage on held-out folds: {summary['cv']['interval_coverage_80']:.0%}.",
        "",
        "## All held-out rows (baselines vs LightGBM)",
        "",
        _markdown(tables["all_rows"]),
        "",
        "## Daily origins where SARIMAX was also run (all four models, same rows)",
        "",
        _markdown(tables["sarimax_subset"]),
        "",
        "## By city (all horizons)",
        "",
        _markdown(tables["by_city"]),
        "",
    ]
    (report_dir / "results.md").write_text("\n".join(lines))
    plot_error_by_horizon(cv, report_dir / "error_by_horizon.png")
    for city in list(cities)[:3]:
        slug = city.lower().replace(" ", "_")
        plot_predictions(cv, city, 24, report_dir / f"predictions_{slug}_24h.png")


def _markdown(table: pd.DataFrame) -> str:
    """A GitHub-flavoured markdown table without the optional `tabulate` dependency."""
    t = table.copy()
    for col in t.select_dtypes("float").columns:
        fmt = "{:.3f}" if str(col).startswith("skill") else "{:.2f}"
        t[col] = t[col].map(fmt.format)
    header = "| " + " | ".join(t.columns) + " |"
    sep = "|" + "|".join("---" for _ in t.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def provider_comparison(
    model: LgbmForecaster,
    cities: Mapping[str, CityData],
    snapshots: pd.DataFrame,
    *,
    trained_until: pd.Timestamp,
) -> pd.DataFrame:
    """Score saved CAMS forecasts against our model on the same issue times.

    `snapshots` has columns city, issued_at, valid_time, pm2_5 (see
    scripts/snapshot_forecasts.py). Only snapshots issued after `trained_until` count, so
    every compared forecast is out-of-sample for our model. For each one, our model
    forecasts from the data that existed at `issued_at`. Returns aligned rows (one per
    city, issue time and valid time); score them with `score_provider_rows`.
    """
    forecaster = Forecaster(model)
    issued_all = pd.to_datetime(snapshots["issued_at"], utc=True)
    snaps = snapshots.assign(
        issued_at=issued_all, valid_time=pd.to_datetime(snapshots["valid_time"], utc=True)
    )[issued_all > trained_until]
    rows = []
    for (city, issued), group in snaps.groupby(["city", "issued_at"]):
        if city not in cities:
            continue
        data = cities[str(city)]
        hourly = data.hourly
        pollutant_cols = hourly.columns.intersection(list(POLLUTANT_COLUMNS))
        origin = pd.Timestamp(str(issued)).floor("h")
        history = hourly.loc[hourly.index <= origin, pollutant_cols]
        weather = hourly.drop(columns=pollutant_cols)
        try:
            fc = forecaster.forecast(history, weather, data.location.timezone)
        except ForecastUnavailableError:
            continue
        ours = pd.DataFrame(
            {"issued_at": issued, "valid_time": fc.frame.index, "pm2_5": fc.frame["pm2_5"]}
        )
        aligned = align_provider_rows(group, hourly["pm2_5"], ours.reset_index(drop=True))
        rows.append(aligned.assign(city=str(city)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
