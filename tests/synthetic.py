"""Synthetic hourly air quality + weather with known structure, for ML tests."""

import numpy as np
import pandas as pd


def city_frame(
    days: int = 400, *, seed: int = 0, level: float = 40.0, start: str = "2023-01-01"
) -> pd.DataFrame:
    """PM2.5 = level * exp(AR(1) noise + diurnal cycle - wind effect), plus co-pollutants."""
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    hour = idx.hour.to_numpy()
    wind = np.clip(3 + np.sin(np.arange(n) / 37) + rng.normal(0, 0.8, n), 0.2, None)
    blh = 400 + 600 * np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) + rng.normal(0, 50, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.95 * noise[i - 1] + rng.normal(0, 0.08)
    diurnal = 0.35 * np.cos(2 * np.pi * (hour - 21) / 24)
    pm = level * np.exp(noise + diurnal - 0.12 * (wind - 3))
    return pd.DataFrame(
        {
            "pm2_5": pm,
            "pm10": pm * 1.7,
            "no2": 25 + 10 * np.cos(2 * np.pi * (hour - 20) / 24) + rng.normal(0, 2, n),
            "o3": np.clip(50 + 40 * np.sin(np.pi * (hour - 8) / 12), 1, None),
            "so2": 8 + rng.normal(0, 1, n).clip(-7, None),
            "co": 400 + pm * 5,
            "temperature_2m": 20 + 6 * np.sin(2 * np.pi * (hour - 9) / 24),
            "relative_humidity_2m": 60 - 15 * np.sin(2 * np.pi * (hour - 9) / 24),
            "precipitation": 0.0,
            "wind_speed_10m": wind,
            "wind_direction_10m": (np.arange(n) * 7) % 360,
            "surface_pressure": 1010.0,
            "cloud_cover": 20.0,
            "boundary_layer_height": blh,
        },
        index=idx,
    )
