"""Application settings, read from environment variables.

Every setting has a working default, so the app runs with no configuration at all.
A `.env` file in the working directory is loaded if present; real environment
variables always win over it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

AqiStandard = Literal["us_epa", "in_naqi"]
LogFormat = Literal["json", "text"]

_VALID_STANDARDS: tuple[AqiStandard, ...] = ("us_epa", "in_naqi")
_VALID_LOG_FORMATS: tuple[LogFormat, ...] = ("json", "text")


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


@dataclass(frozen=True, slots=True)
class Settings:
    default_city: str = "New Delhi"
    aqi_standard: AqiStandard = "us_epa"
    cache_path: Path = Path(".cache/aqd.sqlite3")
    cache_ttl_forecast_s: int = 3600
    cache_ttl_history_s: int = 6 * 3600
    cache_ttl_geocode_s: int = 30 * 86400
    http_timeout_s: float = 10.0
    log_level: str = "INFO"
    log_format: LogFormat = "json"
    model_dir: Path = Path("models")
    # Kept out of repr so the key never lands in logs or tracebacks.
    openaq_api_key: str | None = field(default=None, repr=False)

    @property
    def openaq_enabled(self) -> bool:
        return bool(self.openaq_api_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        d = cls()
        standard = e.get("AQD_AQI_STANDARD", d.aqi_standard).strip().lower()
        if standard not in _VALID_STANDARDS:
            raise ConfigError(
                f"AQD_AQI_STANDARD must be one of {_VALID_STANDARDS}, got {standard!r}"
            )
        log_format = e.get("AQD_LOG_FORMAT", d.log_format).strip().lower()
        if log_format not in _VALID_LOG_FORMATS:
            raise ConfigError(
                f"AQD_LOG_FORMAT must be one of {_VALID_LOG_FORMATS}, got {log_format!r}"
            )
        return cls(
            default_city=e.get("AQD_DEFAULT_CITY", d.default_city).strip() or d.default_city,
            aqi_standard=standard,
            cache_path=Path(e.get("AQD_CACHE_PATH", str(d.cache_path))),
            cache_ttl_forecast_s=_int(e, "AQD_CACHE_TTL_FORECAST", d.cache_ttl_forecast_s),
            cache_ttl_history_s=_int(e, "AQD_CACHE_TTL_HISTORY", d.cache_ttl_history_s),
            cache_ttl_geocode_s=_int(e, "AQD_CACHE_TTL_GEOCODE", d.cache_ttl_geocode_s),
            http_timeout_s=_float(e, "AQD_HTTP_TIMEOUT", d.http_timeout_s),
            log_level=e.get("AQD_LOG_LEVEL", d.log_level).strip().upper(),
            log_format=log_format,
            model_dir=Path(e.get("AQD_MODEL_DIR", str(d.model_dir))),
            openaq_api_key=(e.get("OPENAQ_API_KEY") or "").strip() or None,
        )


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{key} must be >= 0, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{key} must be > 0, got {value}")
    return value


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines from `path` into os.environ without overriding existing vars.

    A ten-line parser instead of python-dotenv: we only need plain KEY=VALUE lines.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
