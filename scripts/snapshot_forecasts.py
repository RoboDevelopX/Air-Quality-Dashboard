"""Save the provider's current PM2.5 forecast for the training cities.

Open-Meteo doesn't archive past air-quality forecasts, so to compare our model against the
provider's forecast on held-out data we keep our own record: each run writes one small
gzip CSV to snapshots/ (city, issued_at, valid_time, pm2_5 for the next 48 hours).
.github/workflows/snapshot.yml runs this twice a day.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from aqd.ml.dataset import TRAINING_CITIES
from aqd.ml.features import MAX_HORIZON
from aqd.providers.http import HttpClient
from aqd.providers.open_meteo import OpenMeteoAirQuality

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    provider = OpenMeteoAirQuality(HttpClient(None, timeout_s=60))
    rows = []
    for loc in TRAINING_CITIES:
        data = provider.air_quality(loc, past_days=0, forecast_days=3)
        fc = data.forecast["pm2_5"].dropna().iloc[:MAX_HORIZON]
        issued = pd.Timestamp(data.issued_at).floor("h")
        rows.append(
            pd.DataFrame(
                {
                    "city": loc.name,
                    "issued_at": issued,
                    "valid_time": fc.index,
                    "pm2_5": fc.round(2),
                }
            )
        )
    out_dir = ROOT / "snapshots"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{datetime.now(UTC):%Y%m%dT%H%M}.csv.gz"
    pd.concat(rows).to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
