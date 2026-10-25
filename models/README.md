# Model artifact

`scripts/train.py` writes the trained model here and the app loads it at startup:

- `pm25_lgbm.txt`: LightGBM model in LightGBM's own text format (no pickle, so loading it can't run code)
- `pm25_lgbm.json`: feature names, per-horizon interval offsets and the evaluation summary

If these files are missing, the app still runs and shows the provider's forecast only.
