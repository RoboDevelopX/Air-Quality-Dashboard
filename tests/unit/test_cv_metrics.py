import itertools

import numpy as np
import pandas as pd
import pytest

from aqd.ml.cv import make_folds, split
from aqd.ml.features import build_frame
from aqd.ml.metrics import bucket_of, coverage, mae, rmse, score_table, skill
from tests.synthetic import city_frame


def test_folds_tile_recent_period_with_gap() -> None:
    idx = pd.date_range("2023-01-01", periods=400 * 24, freq="h", tz="UTC")
    folds = make_folds(idx, n_folds=3, test_days=30, gap_hours=48, min_train_days=180)
    assert [f.number for f in folds] == [1, 2, 3]
    assert folds[-1].test_end == idx.max() + pd.Timedelta(hours=1)
    for a, b in itertools.pairwise(folds):
        assert a.test_end == b.test_start
    for f in folds:
        assert f.test_start - f.train_end == pd.Timedelta(hours=49)


def test_not_enough_history_raises() -> None:
    idx = pd.date_range("2023-01-01", periods=60 * 24, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="not enough history"):
        make_folds(idx, n_folds=3, test_days=30)
    with pytest.raises(ValueError):
        make_folds(pd.DatetimeIndex([]))


def test_split_never_trains_on_test_period_targets() -> None:
    df = city_frame(days=250)
    origins = df.index[df.index.hour % 6 == 0]
    frame = build_frame(df, "UTC", origins=origins)
    folds = make_folds(origins, n_folds=2, test_days=20, gap_hours=48, min_train_days=150)
    for fold in folds:
        train, test = split(frame, fold)
        assert train.any() and test.any()
        assert not (train & test).any()
        assert frame.loc[train, "valid_time"].max() < fold.test_start
        assert frame.loc[test, "origin"].min() >= fold.test_start


def test_split_enforces_target_cutoff_even_with_small_gap() -> None:
    df = city_frame(days=250)
    frame = build_frame(df, "UTC", origins=df.index[df.index.hour == 0])
    fold = make_folds(df.index, n_folds=1, test_days=20, gap_hours=0, min_train_days=150)[0]
    train, _ = split(frame, fold)
    assert frame.loc[train, "valid_time"].max() < fold.test_start


def test_metric_values() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, np.array([2.0, 2.0, 2.0])) == pytest.approx(2 / 3)
    assert rmse(y, np.array([1.0, 2.0, 5.0])) == pytest.approx(np.sqrt(4 / 3))
    assert skill(1.0, 2.0) == 0.5
    assert np.isnan(skill(1.0, 0.0))
    assert coverage(y, y - 1, y + 1) == 1.0
    assert coverage(y, y + 1, y + 2) == 0.0


def test_bucket_of() -> None:
    assert bucket_of(pd.Series([1, 6, 7, 24, 25, 48])).tolist() == [
        "1-6h",
        "1-6h",
        "7-24h",
        "7-24h",
        "25-48h",
        "25-48h",
    ]


def test_score_table_groups_and_skill() -> None:
    frame = pd.DataFrame(
        {
            "y": [10.0, 10.0, 10.0, np.nan],
            "horizon": [1, 10, 30, 1],
            "persistence": [12.0, 14.0, 16.0, 1.0],
            "model": [11.0, 12.0, 13.0, 1.0],
        }
    )
    table = score_table(frame, {"persistence": "persistence", "model": "model"})
    overall = table[(table["group"] == "all") & (table["model"] == "model")].iloc[0]
    assert overall["n"] == 3  # NaN target dropped
    assert overall["mae"] == pytest.approx(2.0)
    assert overall["skill_vs_persistence"] == pytest.approx(0.5)
    assert set(table["group"]) == {"1-6h", "7-24h", "25-48h", "all"}
    flat = score_table(frame, {"persistence": "persistence"}, by=None)
    assert flat["group"].tolist() == ["all"]
