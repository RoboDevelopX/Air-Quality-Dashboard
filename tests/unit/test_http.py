from pathlib import Path

import httpx
import pytest
import respx

from aqd.providers.http import HttpClient, ProviderError
from aqd.storage.cache import SqliteCache
from tests.conftest import FakeClock

URL = "https://api.example.test/v1/data"


def client(tmp_path: Path, clock: FakeClock, **kw: object) -> HttpClient:
    cache = SqliteCache(tmp_path / "c.sqlite3", clock=clock)
    return HttpClient(cache, sleep=lambda _: None, clock=clock, **kw)  # type: ignore[arg-type]


@respx.mock
def test_fetches_then_serves_from_cache(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"v": 1}))
    http = client(tmp_path, clock)
    first = http.get_json(URL, {"a": 1}, ttl_s=60)
    second = http.get_json(URL, {"a": 1}, ttl_s=60)
    assert first.data == second.data == {"v": 1}
    assert not first.from_cache and second.from_cache
    assert route.call_count == 1
    assert first.fetched_at == clock.now


@respx.mock
def test_refetches_after_ttl(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(200, json={"v": 1}), httpx.Response(200, json={"v": 2})]
    )
    http = client(tmp_path, clock)
    http.get_json(URL, {}, ttl_s=60)
    clock.advance(61)
    assert http.get_json(URL, {}, ttl_s=60).data == {"v": 2}
    assert route.call_count == 2


@respx.mock
def test_retries_transient_errors(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("boom"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleeps: list[float] = []
    http = HttpClient(None, sleep=sleeps.append, retries=2, backoff_s=0.5)
    assert http.get_json(URL, {}, ttl_s=60).data == {"ok": True}
    assert route.call_count == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff


@respx.mock
def test_client_errors_are_not_retried(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(
        return_value=httpx.Response(400, json={"error": True, "reason": "bad variable"})
    )
    with pytest.raises(ProviderError, match="HTTP 400: bad variable"):
        client(tmp_path, clock).get_json(URL, {}, ttl_s=60)
    assert route.call_count == 1


@respx.mock
def test_gives_up_after_retries(tmp_path: Path, clock: FakeClock) -> None:
    respx.get(URL).mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(ProviderError, match="after 3 attempts: HTTP 429: slow down"):
        client(tmp_path, clock).get_json(URL, {}, ttl_s=60)


@respx.mock
def test_stale_cache_fallback_on_failure(tmp_path: Path, clock: FakeClock) -> None:
    respx.get(URL).mock(
        side_effect=[httpx.Response(200, json={"v": 1}), httpx.Response(500), httpx.Response(500)]
    )
    http = client(tmp_path, clock, retries=1)
    http.get_json(URL, {}, ttl_s=60)
    clock.advance(3600)
    result = http.get_json(URL, {}, ttl_s=60)
    assert result.data == {"v": 1}
    assert result.stale and result.from_cache


@respx.mock
def test_invalid_json_raises_after_retries(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, text="<html>"))
    with pytest.raises(ProviderError, match="invalid JSON"):
        client(tmp_path, clock).get_json(URL, {}, ttl_s=60)
    assert route.call_count == 3


@respx.mock
def test_invalid_json_is_retried(tmp_path: Path, clock: FakeClock) -> None:
    respx.get(URL).mock(
        side_effect=[httpx.Response(200, text='{"cut'), httpx.Response(200, json={"ok": 1})]
    )
    assert client(tmp_path, clock).get_json(URL, {}, ttl_s=60).data == {"ok": 1}


@respx.mock
def test_headers_sent_but_not_part_of_cache_key(tmp_path: Path, clock: FakeClock) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={}))
    http = client(tmp_path, clock)
    http.get_json(URL, {}, ttl_s=60, headers={"X-API-Key": "k1"})
    assert route.calls[0].request.headers["X-API-Key"] == "k1"
    http.get_json(URL, {}, ttl_s=60, headers={"X-API-Key": "k2"})
    assert route.call_count == 1
    http.close()


@respx.mock
def test_error_reason_fallbacks(tmp_path: Path, clock: FakeClock) -> None:
    respx.get(URL).mock(return_value=httpx.Response(404, json={"detail": "not found"}))
    with pytest.raises(ProviderError, match="not found"):
        client(tmp_path, clock).get_json(URL, {"x": 1}, ttl_s=60)
    respx.get(URL).mock(return_value=httpx.Response(404, json=["odd"]))
    with pytest.raises(ProviderError, match="odd"):
        client(tmp_path, clock).get_json(URL, {"x": 2}, ttl_s=60)
