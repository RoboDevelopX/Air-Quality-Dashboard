# Test fixtures

Raw API response bodies used by the provider tests (via respx). Tests never call the network.

The Open-Meteo fixtures were recorded from the live APIs on 2026-10-25 (Delhi, hours from
2026-10-23T00:00Z). The OpenAQ fixtures are still hand-written from the v3 docs because no
API key was used; re-record them with `--openaq` once you have one.

Re-record from the live APIs with:

```bash
python scripts/record_fixtures.py            # Open-Meteo only
OPENAQ_API_KEY=... python scripts/record_fixtures.py --openaq
```

The recorder writes the response body only; request headers (and so API keys) are never saved.

| File | Request |
|---|---|
| `open_meteo_air_quality.json` | Air quality, Delhi, 6 pollutants, `past_days=2`, `forecast_days=2`, `timeformat=unixtime` |
| `open_meteo_weather.json` | Forecast API, Delhi, weather variables, same window |
| `open_meteo_geocoding.json` | Geocoding search `name=Delhi`, `count=2` |
| `open_meteo_geocoding_empty.json` | Geocoding search with no matches |
| `open_meteo_error_400.json` | Air quality request with an invalid variable |
| `openaq_locations.json` | OpenAQ v3 locations within 25 km of Delhi |
| `openaq_sensor_hours.json` | OpenAQ v3 hourly values for one PM2.5 sensor |
