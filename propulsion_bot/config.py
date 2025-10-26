"""Configuration helpers for the Propulsion bot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables once on import so every module that relies on the
# configuration picks up settings from ``.env`` when present.
load_dotenv()

# Minimal embedded configuration used when ``config/strategy.yaml`` is not
# available.  Runtime callers rely on ``Settings.get`` defaults for most values
# so an empty dictionary is sufficient as a safe fallback.
DEFAULT_CONFIG = "{}"


class Settings:
    """Singleton wrapper for accessing configuration values."""

    _instance: "Settings | None" = None

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        cfg_path = Path("config") / "strategy.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as handle:
                self.config = yaml.safe_load(handle) or {}
        else:
            self.config = yaml.safe_load(DEFAULT_CONFIG)

        self.ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.ibkr_port = int(os.getenv("IBKR_PORT", "7497"))
        self.ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))
        self.finnhub_token = os.getenv("FINNHUB_TOKEN", "")
        self.dashboard_port = int(os.getenv("DASHBOARD_PORT", "8501"))
        self.premarket_db = os.getenv(
            "PREMARKET_DB", str(Path("..") / "bot1.1" / "premarket.db")
        )
        self.force_market_open = str(os.getenv("FORCE_MARKET_OPEN", "false")).lower() in (
            "1",
            "true",
            "yes",
        )

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self.config
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


__all__ = ["Settings", "DEFAULT_CONFIG"]

