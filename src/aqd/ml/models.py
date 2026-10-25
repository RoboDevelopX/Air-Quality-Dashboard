"""Forecasting models: a global LightGBM direct multi-horizon model and per-series SARIMAX.

The LightGBM artifact is saved as LightGBM's own text format plus a JSON sidecar, not a
pickle: loading it can't execute code, and the files diff readably in git.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aqd.ml.features import MAX_HORIZON, feature_columns, invert_delta, target_delta

log = logging.getLogger(__name__)

MODEL_FILE = "pm25_lgbm.txt"
META_FILE = "pm25_lgbm.json"
INTERVAL_QUANTILES = (0.1, 0.9)  # 80% interval

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "l1",  # median in log-ratio space: robust to spikes, matches MAE
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "num_threads": 0,
}


@dataclass
class LgbmForecaster:
    booster: lgb.Booster
    features: list[str]
    # Per horizon: (low, high) quantiles of out-of-fold residuals in log-ratio space.
    intervals: dict[int, tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 600,
        early_stopping_frac: float = 0.1,
    ) -> LgbmForecaster:
        """Fit on a stacked frame. The last `early_stopping_frac` of origins (by time) is held
        out for early stopping, so the stopping point is never chosen on the future."""
        data = frame.dropna(subset=["y", "pm_log_0"])
        if data.empty:
            raise ValueError("no training rows with a target")
        features = feature_columns(data)
        y = target_delta(data)
        cutoff = data["origin"].quantile(1 - early_stopping_frac)
        fit_mask = (data["origin"] < cutoff).to_numpy()
        p = {**DEFAULT_PARAMS, **(params or {})}
        if early_stopping_frac > 0 and fit_mask.sum() > 0 and (~fit_mask).sum() > 0:
            train_set = lgb.Dataset(data.loc[fit_mask, features], y[fit_mask])
            valid_set = lgb.Dataset(data.loc[~fit_mask, features], y[~fit_mask])
            booster = lgb.train(
                p,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            best = booster.best_iteration or num_boost_round
            # Refit on everything with the chosen number of rounds.
            booster = lgb.train(p, lgb.Dataset(data[features], y), num_boost_round=best)
        else:
            booster = lgb.train(p, lgb.Dataset(data[features], y), num_boost_round=num_boost_round)
        return cls(booster=booster, features=features)

    def predict_delta(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(frame[self.features]), dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """PM2.5 forecast in µg/m³ for each row."""
        return invert_delta(frame, self.predict_delta(frame))

    def calibrate(self, frame: pd.DataFrame, delta_pred: np.ndarray) -> None:
        """Set per-horizon interval offsets from out-of-fold residuals (split conformal)."""
        resid = (target_delta(frame) - delta_pred).to_numpy()
        horizons = frame["horizon"].to_numpy()
        lo_q, hi_q = INTERVAL_QUANTILES
        self.intervals = {}
        for h in np.unique(horizons):
            r = resid[(horizons == h) & ~np.isnan(resid)]
            if len(r) >= 20:
                self.intervals[int(h)] = (float(np.quantile(r, lo_q)), float(np.quantile(r, hi_q)))

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """80% interval bounds in µg/m³. NaN where a horizon was never calibrated."""
        delta = self.predict_delta(frame)
        offsets = np.array(
            [self.intervals.get(int(h), (np.nan, np.nan)) for h in frame["horizon"]], dtype=float
        ).reshape(-1, 2)
        lo = invert_delta(frame, delta + offsets[:, 0])
        hi = invert_delta(frame, delta + offsets[:, 1])
        return lo, hi

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / MODEL_FILE))
        meta = {
            "features": self.features,
            "intervals": {str(h): list(v) for h, v in sorted(self.intervals.items())},
            "interval_quantiles": list(INTERVAL_QUANTILES),
            "max_horizon": MAX_HORIZON,
            **self.metadata,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, default=str) + "\n")

    @classmethod
    def load(cls, directory: Path) -> LgbmForecaster:
        model_path, meta_path = directory / MODEL_FILE, directory / META_FILE
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"no model artifact in {directory}")
        meta = json.loads(meta_path.read_text())
        booster = lgb.Booster(model_file=str(model_path))
        intervals = {int(h): (float(v[0]), float(v[1])) for h, v in meta.pop("intervals").items()}
        features = list(meta.pop("features"))
        if booster.feature_name() != features:
            raise ValueError("model features do not match metadata")
        return cls(booster=booster, features=features, intervals=intervals, metadata=meta)


def sarimax_forecast(
    history: pd.Series,
    steps: int = MAX_HORIZON,
    *,
    window_hours: int = 28 * 24,
    order: tuple[int, int, int] = (2, 0, 1),
) -> np.ndarray:
    """Forecast PM2.5 `steps` hours ahead with SARIMAX on log1p values.

    Daily seasonality enters as Fourier exogenous terms (k=2) rather than a seasonal order
    of 24, which would make each fit ~50x slower for little gain. Fitted per origin on a
    rolling window, so it's evaluated offline only. Falls back to persistence if the fit
    fails or the window has too little data.
    """
    # Training-only dependency, imported lazily so the app doesn't need statsmodels.
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = history.iloc[-window_hours:]
    last = float(hist.dropna().iloc[-1]) if hist.notna().any() else float("nan")
    if hist.notna().sum() < 7 * 24:
        return np.full(steps, last)
    index = pd.DatetimeIndex(hist.index)
    future = pd.date_range(index[-1] + pd.Timedelta(hours=1), periods=steps, freq="h")
    exog_hist, exog_future = _fourier(index), _fourier(future)
    y = np.log1p(hist.clip(lower=0).to_numpy())
    try:
        model = SARIMAX(
            y,
            exog=exog_hist,
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            # maxiter is capped for speed; a not-fully-converged fit is still a usable baseline.
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = model.fit(disp=False, maxiter=50)
        fc = np.expm1(result.forecast(steps=steps, exog=exog_future))
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.warning("SARIMAX failed, using persistence", extra={"error": str(exc)})
        return np.full(steps, last)
    clipped: np.ndarray = np.clip(np.asarray(fc, dtype=float), 0, None)
    return clipped


def _fourier(index: pd.DatetimeIndex, k: int = 2) -> np.ndarray:
    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60
    cols = []
    for i in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * i * hours / 24))
        cols.append(np.cos(2 * np.pi * i * hours / 24))
    out: np.ndarray = np.column_stack(cols)
    return out
