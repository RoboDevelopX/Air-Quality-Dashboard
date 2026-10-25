"""Run the saved model on live data: the next 48 hours of PM2.5 from the latest hour."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aqd.ml.features import MAX_HORIZON, build_frame
from aqd.ml.models import LgbmForecaster

log = logging.getLogger(__name__)

MIN_HISTORY_HOURS = 72


class ForecastUnavailableError(RuntimeError):
    """Not enough recent data to forecast."""


@dataclass(frozen=True, slots=True)
class ModelForecast:
    origin: pd.Timestamp
    frame: pd.DataFrame  # index valid_time (UTC); columns pm2_5, lo, hi, horizon
    metadata: dict[str, Any]


class Forecaster:
    """Thin wrapper the app uses; loads the artifact once and predicts per request."""

    def __init__(self, model: LgbmForecaster) -> None:
        self.model = model

    @classmethod
    def load(cls, model_dir: Path) -> Forecaster:
        return cls(LgbmForecaster.load(model_dir))

    @property
    def metadata(self) -> dict[str, Any]:
        return self.model.metadata

    def forecast(
        self,
        history: pd.DataFrame,
        weather: pd.DataFrame | None,
        tz: str,
        horizons: int = MAX_HORIZON,
    ) -> ModelForecast:
        """Forecast from the last hour with a PM2.5 value in `history`.

        `history`: hourly pollutants (µg/m³, UTC index) up to now.
        `weather`: hourly weather covering the history and the next `horizons` hours (the
        future part is the weather forecast), or None to forecast without weather.
        """
        pm = history["pm2_5"].dropna() if "pm2_5" in history else pd.Series(dtype=float)
        if len(pm) < MIN_HISTORY_HOURS:
            raise ForecastUnavailableError(
                f"need at least {MIN_HISTORY_HOURS} hours of PM2.5 history, have {len(pm)}"
            )
        origin = pd.Timestamp(pm.index.max())
        hist = history[history.index <= origin]
        future_index = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=horizons, freq="h", tz="UTC"
        )
        frame = pd.concat([hist, pd.DataFrame(index=future_index)])
        if weather is not None and not weather.empty:
            frame = frame.join(weather[~weather.index.duplicated()], how="left")
        rows = build_frame(
            frame, tz, horizons=range(1, horizons + 1), origins=pd.DatetimeIndex([origin])
        )
        point = self.model.predict(rows)
        lo, hi = self.model.predict_interval(rows)
        out = pd.DataFrame(
            {"pm2_5": point, "lo": lo, "hi": hi, "horizon": rows["horizon"].to_numpy()},
            index=pd.DatetimeIndex(rows["valid_time"], name="time"),
        )
        return ModelForecast(origin=origin, frame=out, metadata=self.model.metadata)


def load_forecaster(model_dir: Path) -> Forecaster | None:
    """The saved model, or None if it can't be used; the app then shows CAMS's forecast only."""
    try:
        return Forecaster.load(model_dir)
    except FileNotFoundError:
        log.warning("no model artifact found", extra={"model_dir": str(model_dir)})
    except (ImportError, OSError) as exc:
        # LightGBM's native library didn't load. On macOS that means libomp is missing.
        log.warning(
            "LightGBM could not be loaded; on macOS run `brew install libomp`",
            extra={"error": str(exc)},
        )
    return None
