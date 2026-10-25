import json
import logging

import pytest

from aqd.log import configure_logging


def test_json_logs_include_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    extra = {"provider": "open_meteo", "latency_ms": 12}
    logging.getLogger("aqd.test").info("fetched", extra=extra)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["msg"] == "fetched"
    assert record["provider"] == "open_meteo"
    assert record["latency_ms"] == 12
    assert record["level"] == "INFO"


def test_json_logs_exceptions(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging.getLogger("aqd.test").exception("failed")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "RuntimeError: boom" in record["exc"]


def test_text_format_and_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "text")
    log = logging.getLogger("aqd.test")
    log.info("hidden")
    log.warning("shown")
    out = capsys.readouterr().out
    assert "hidden" not in out
    assert "WARNING aqd.test: shown" in out
