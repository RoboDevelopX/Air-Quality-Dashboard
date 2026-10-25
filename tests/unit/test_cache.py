from pathlib import Path

from aqd.storage.cache import SqliteCache, make_key
from tests.conftest import FakeClock


def test_make_key_ignores_param_order() -> None:
    assert make_key("u", {"a": 1, "b": 2}) == make_key("u", {"b": 2, "a": 1})
    assert make_key("u", {"a": 1}) != make_key("u", {"a": 2})
    assert make_key("u") == make_key("u", {})


def test_set_get_and_ttl_expiry(tmp_path: Path, clock: FakeClock) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite3", clock=clock)
    assert cache.get("k") is None
    cache.set("k", "https://x", '{"a": 1}', ttl_s=60)
    entry = cache.get("k")
    assert entry is not None
    assert entry.body == '{"a": 1}'
    assert entry.is_fresh
    clock.advance(59)
    assert cache.get("k").is_fresh  # type: ignore[union-attr]
    clock.advance(1)
    stale = cache.get("k")
    assert stale is not None  # kept for stale fallback
    assert not stale.is_fresh
    assert stale.age_s == 60


def test_overwrite_refreshes(tmp_path: Path, clock: FakeClock) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite3", clock=clock)
    cache.set("k", "u", "old", 10)
    clock.advance(100)
    cache.set("k", "u", "new", 10)
    entry = cache.get("k")
    assert entry is not None and entry.body == "new" and entry.is_fresh
    assert len(cache) == 1


def test_purge_removes_only_old_entries(tmp_path: Path, clock: FakeClock) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite3", clock=clock)
    cache.set("old", "u", "1", 10)
    clock.advance(1000)
    cache.set("new", "u", "2", 10)
    assert cache.purge(max_age_s=500) == 1
    assert cache.get("old") is None
    assert cache.get("new") is not None


def test_persists_across_instances(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "nested" / "c.sqlite3"
    SqliteCache(path, clock=clock).set("k", "u", "v", 10)
    assert SqliteCache(path, clock=clock).get("k") is not None
