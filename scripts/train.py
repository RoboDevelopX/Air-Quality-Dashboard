"""Fetch training data, evaluate all models with time-series CV, and save the artifact.

Usage:
    python scripts/train.py                     # full run: 10 cities since Aug 2022
    python scripts/train.py --cities Delhi London --folds 3 --no-sarimax   # quick run

Outputs:
    models/pm25_lgbm.txt, models/pm25_lgbm.json   model artifact loaded by the app
    reports/metrics.json, reports/results.md      CV metrics (baselines vs models)
    reports/*.png                                 evaluation plots
The comparison with the provider's own forecast is scripts/compare_provider.py.
Raw data is cached under data/ (gitignored) so reruns don't refetch.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from aqd.config import Settings, load_dotenv
from aqd.log import configure_logging
from aqd.ml.dataset import DATA_START, TRAINING_CITIES, CityData, fetch_city
from aqd.ml.train import TrainConfig, train_pipeline
from aqd.providers.http import HttpClient
from aqd.providers.open_meteo import OpenMeteoAirQuality, OpenMeteoWeather
from aqd.storage.cache import SqliteCache

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("aqd.train")


def load_cities(
    names: list[str] | None, start: date, end: date, http: HttpClient
) -> dict[str, CityData]:
    aq = OpenMeteoAirQuality(http, ttl_history_s=30 * 86400)
    weather = OpenMeteoWeather(http, ttl_history_s=30 * 86400)
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for loc in TRAINING_CITIES:
        if names and loc.name not in names:
            continue
        path = raw_dir / f"{loc.name.lower().replace(' ', '_')}_{start}_{end}.parquet"
        if path.exists():
            hourly = pd.read_parquet(path)
        else:
            log.info("fetching", extra={"city": loc.name, "start": str(start), "end": str(end)})
            hourly = fetch_city(aq, weather, loc, start, end).hourly
            hourly.to_parquet(path)
        out[loc.name] = CityData(loc, hourly)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--cities", nargs="*", help="subset of training city names")
    parser.add_argument("--start", type=date.fromisoformat, default=DATA_START)
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="default: 2 days ago")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--no-sarimax", action="store_true", help="skip SARIMAX (much faster)")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings.log_level, "text")
    end = args.end or (datetime.now(UTC).date() - timedelta(days=2))
    # Long, one-off requests: be patient with Open-Meteo's rate limiting and dropped connections.
    http = HttpClient(
        SqliteCache(ROOT / "data" / "http_cache.sqlite3"), timeout_s=120, retries=5, backoff_s=2.0
    )

    cities = load_cities(args.cities, args.start, end, http)
    config = TrainConfig(
        n_folds=args.folds,
        test_days=args.test_days,
        sarimax_every_days=10**6 if args.no_sarimax else 1,
        lgbm_rounds=args.rounds,
    )
    train_pipeline(cities, model_dir=args.model_dir, report_dir=args.report_dir, config=config)
    print((args.report_dir / "results.md").read_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
