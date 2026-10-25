"""Shared HTTP client: timeouts, retries with backoff, caching and stale fallback."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from aqd.storage.cache import SqliteCache, make_key

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "aqd-air-quality-dashboard/0.1"


class ProviderError(RuntimeError):
    """A provider request failed and no cached copy was available."""


@dataclass(frozen=True, slots=True)
class FetchResult:
    data: Any
    from_cache: bool
    stale: bool  # True when served from an expired cache entry because the fetch failed
    fetched_at: float


class HttpClient:
    def __init__(
        self,
        cache: SqliteCache | None = None,
        *,
        timeout_s: float = 10.0,
        retries: int = 2,
        backoff_s: float = 0.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache = cache
        self._clock = clock
        self._retries = retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._client = client or httpx.Client(
            timeout=timeout_s, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any],
        *,
        ttl_s: float,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """GET `url` and decode JSON, using the cache when it holds a fresh copy.

        Headers (which may carry API keys) are deliberately not part of the cache key and
        are never logged.
        """
        key = make_key(url, params)
        cached = self._cache.get(key) if self._cache else None
        if cached is not None and cached.is_fresh:
            log.debug("cache hit", extra={"url": url, "age_s": round(cached.age_s)})
            return FetchResult(json.loads(cached.body), True, False, cached.fetched_at)

        try:
            data, body = self._fetch(url, params, headers)
        except ProviderError:
            if cached is not None:
                log.warning(
                    "serving stale cache after fetch failure",
                    extra={"url": url, "age_s": round(cached.age_s)},
                )
                return FetchResult(json.loads(cached.body), True, True, cached.fetched_at)
            raise
        now = self._clock()
        if self._cache is not None:
            self._cache.set(key, url, body, ttl_s)
        return FetchResult(data, False, False, now)

    def _fetch(
        self, url: str, params: Mapping[str, Any], headers: Mapping[str, str] | None
    ) -> tuple[Any, str]:
        last_error = "unknown error"
        for attempt in range(self._retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.get(url, params=dict(params), headers=dict(headers or {}))
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "http transport error",
                    extra={"url": url, "attempt": attempt + 1, "error": last_error},
                )
            else:
                latency_ms = round((time.perf_counter() - started) * 1000)
                log.info(
                    "http request",
                    extra={"url": url, "status": resp.status_code, "latency_ms": latency_ms},
                )
                if resp.status_code == 200:
                    try:
                        return resp.json(), resp.text
                    except ValueError as exc:
                        raise ProviderError(f"invalid JSON from {url}") from exc
                last_error = f"HTTP {resp.status_code}: {_reason(resp)}"
                if resp.status_code not in RETRY_STATUS:
                    raise ProviderError(last_error)
            if attempt < self._retries:
                self._sleep(self._backoff_s * 2**attempt)
        raise ProviderError(
            f"request to {url} failed after {self._retries + 1} attempts: {last_error}"
        )

    def close(self) -> None:
        self._client.close()


def _reason(resp: httpx.Response) -> str:
    """Best-effort error reason; Open-Meteo returns {"error": true, "reason": "..."}."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:200]
    if isinstance(body, dict):
        for key in ("reason", "detail", "message"):
            if key in body:
                return str(body[key])[:200]
    return str(body)[:200]
