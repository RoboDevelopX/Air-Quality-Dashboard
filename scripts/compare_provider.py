"""Compare our saved model with the provider's (CAMS) own forecasts on held-out data.

Usage: python scripts/compare_provider.py

Open-Meteo doesn't archive past air-quality forecasts, so scripts/snapshot_forecasts.py
saves them to snapshots/. This script takes every snapshot issued after the saved model's
training data ends, fetches the CAMS values that have since arrived (and archived weather),
re-runs our model from the data that existed at each issue time, and writes
reports/provider_comparison.md. Run it once a few days of snapshots have their actuals.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from aqd.config import load_dotenv
from aqd.log import configure_logging
from aqd.ml.dataset import TRAINING_CITIES, CityData, fetch_city
from aqd.ml.evaluate import score_provider_rows
from aqd.ml.features import LAGS
from aqd.ml.models import LgbmForecaster
from aqd.ml.train import _markdown, provider_comparison
from aqd.providers.http import HttpClient
from aqd.providers.open_meteo import OpenMeteoAirQuality, OpenMeteoWeather

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("aqd.compare_provider")


def main() -> int:
    load_dotenv()
    configure_logging("INFO", "text")
    model = LgbmForecaster.load(ROOT / "models")
    trained_until = pd.Timestamp(model.metadata["summary"]["data_end"])

    files = sorted((ROOT / "snapshots").glob("*.csv.gz"))
    snapshots = pd.concat(pd.read_csv(f) for f in files) if files else pd.DataFrame()
    if snapshots.empty:
        print("No snapshots yet: run scripts/snapshot_forecasts.py (or let the workflow do it).")
        return 0
    snapshots = snapshots[pd.to_datetime(snapshots["issued_at"], utc=True) > trained_until]
    if snapshots.empty:
        print(f"No snapshots issued after the model's training data ends ({trained_until}).")
        return 0

    # Enough history before the first issue time for the longest lag feature, and every hour
    # up to now. Hours after now are CAMS forecasts, so they're blanked, never used as actuals.
    now = pd.Timestamp(datetime.now(UTC)).floor("h")
    first_issue = pd.to_datetime(snapshots["issued_at"], utc=True).min()
    start = (first_issue - timedelta(hours=max(LAGS) + 72)).date()
    http = HttpClient(None, timeout_s=60, retries=5, backoff_s=2.0)
    aq, weather = OpenMeteoAirQuality(http), OpenMeteoWeather(http)
    cities: dict[str, CityData] = {}
    for loc in TRAINING_CITIES:
        if loc.name in set(snapshots["city"]):
            cities[loc.name] = fetch_city(aq, weather, loc, start, now.date(), observed_until=now)

    rows = provider_comparison(model, cities, snapshots, trained_until=trained_until)
    if rows.empty:
        print("Snapshots exist, but none of their forecast hours have actuals yet.")
        return 0
    table = score_provider_rows(rows)
    issues = rows[["city", "issued_at"]].drop_duplicates().shape[0]
    text = (
        "# Model vs provider (CAMS) forecasts\n\n"
        f"{issues} forecast issues, {len(rows)} hourly comparisons, all issued after the "
        f"model's training data ends ({str(trained_until)[:16]}).\n\n"
        "CAMS is scored against Open-Meteo's archive, which is stitched from the first hours of "
        "each CAMS run. Until the next run (about 12 h) the provider's forecast *is* the "
        "'actual', so its 1-6h error is close to zero by construction; only the 25-48h bucket "
        "is a fair-ish comparison, and it still favours CAMS.\n\n" + _markdown(table) + "\n"
    )
    out = ROOT / "reports" / "provider_comparison.md"
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
