"""SQLite-backed response cache with per-entry TTL.

Why SQLite: it ships with Python, survives app restarts (unlike st.cache_data), and is
safe across Streamlit's script threads when each operation opens its own connection.
Expired entries are kept (not deleted on read) so a provider outage can fall back to
stale data; `purge` removes entries older than a hard limit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    body       TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    ttl_s      REAL NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    body: str
    fetched_at: float
    ttl_s: float
    now: float

    @property
    def age_s(self) -> float:
        return self.now - self.fetched_at

    @property
    def is_fresh(self) -> bool:
        return self.age_s < self.ttl_s


def make_key(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Stable key from URL + params, independent of parameter order."""
    payload = json.dumps({"url": url, "params": dict(params or {})}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class SqliteCache:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def get(self, key: str) -> CacheEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT body, fetched_at, ttl_s FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(body=row[0], fetched_at=row[1], ttl_s=row[2], now=self._clock())

    def set(self, key: str, url: str, body: str, ttl_s: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO http_cache (key, url, body, fetched_at, ttl_s) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, url, body, self._clock(), ttl_s),
            )

    def purge(self, max_age_s: float) -> int:
        """Delete entries fetched more than `max_age_s` ago. Returns rows removed."""
        cutoff = self._clock() - max_age_s
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM http_cache WHERE fetched_at < ?", (cutoff,))
            return cur.rowcount

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0])
