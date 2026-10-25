from pathlib import Path

import pytest

from aqd.config import ConfigError, Settings, load_dotenv


def test_defaults_when_env_empty() -> None:
    s = Settings.from_env({})
    assert s == Settings()
    assert s.openaq_enabled is False


def test_reads_and_normalises_values() -> None:
    s = Settings.from_env(
        {
            "AQD_DEFAULT_CITY": "  Mumbai ",
            "AQD_AQI_STANDARD": "IN_NAQI",
            "AQD_CACHE_TTL_FORECAST": "60",
            "AQD_HTTP_TIMEOUT": "2.5",
            "AQD_LOG_LEVEL": "debug",
            "AQD_LOG_FORMAT": "TEXT",
            "AQD_MODEL_DIR": "/tmp/models",
            "OPENAQ_API_KEY": " secret ",
        }
    )
    assert s.default_city == "Mumbai"
    assert s.aqi_standard == "in_naqi"
    assert s.cache_ttl_forecast_s == 60
    assert s.http_timeout_s == 2.5
    assert s.log_level == "DEBUG"
    assert s.log_format == "text"
    assert s.model_dir == Path("/tmp/models")
    assert s.openaq_enabled is True


def test_api_key_never_in_repr() -> None:
    s = Settings.from_env({"OPENAQ_API_KEY": "super-secret"})
    assert "super-secret" not in repr(s)


def test_blank_values_fall_back_to_defaults() -> None:
    env = {"AQD_CACHE_TTL_HISTORY": " ", "AQD_DEFAULT_CITY": "", "OPENAQ_API_KEY": ""}
    s = Settings.from_env(env)
    assert s.cache_ttl_history_s == Settings().cache_ttl_history_s
    assert s.default_city == Settings().default_city
    assert s.openaq_api_key is None


@pytest.mark.parametrize(
    "env",
    [
        {"AQD_AQI_STANDARD": "eu"},
        {"AQD_LOG_FORMAT": "xml"},
        {"AQD_CACHE_TTL_FORECAST": "abc"},
        {"AQD_CACHE_TTL_FORECAST": "-1"},
        {"AQD_HTTP_TIMEOUT": "0"},
        {"AQD_HTTP_TIMEOUT": "fast"},
    ],
)
def test_invalid_values_raise(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_load_dotenv_does_not_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nAQD_X_NEW='from-file'\nAQD_X_SET=from-file\nnot a pair\n")
    monkeypatch.setenv("AQD_X_SET", "from-env")
    monkeypatch.delenv("AQD_X_NEW", raising=False)
    load_dotenv(env_file)
    import os

    assert os.environ["AQD_X_NEW"] == "from-file"
    assert os.environ["AQD_X_SET"] == "from-env"
    monkeypatch.delenv("AQD_X_NEW")


def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "nope.env")
