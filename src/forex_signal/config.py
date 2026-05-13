"""Configuration loading from config.yaml and .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class MT5Credentials:
    login: int
    password: str
    server: str
    path: str = ""


@dataclass
class AppConfig:
    raw: dict[str, Any]
    symbol: str
    timeframe: str
    mt5: MT5Credentials | None = None
    project_root: Path = field(default=PROJECT_ROOT)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(
    config_path: Path | str | None = None,
    env_path: Path | str | None = None,
) -> AppConfig:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    env_p = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if env_p.exists():
        load_dotenv(env_p, override=False)

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    mt5_creds = None
    login_str = os.getenv("MT5_LOGIN", "").strip()
    if login_str:
        try:
            mt5_creds = MT5Credentials(
                login=int(login_str),
                password=os.getenv("MT5_PASSWORD", ""),
                server=os.getenv("MT5_SERVER", ""),
                path=os.getenv("MT5_PATH", ""),
            )
        except ValueError:
            mt5_creds = None

    return AppConfig(
        raw=raw,
        symbol=os.getenv("SYMBOL", raw.get("symbol", "EURUSD")),
        timeframe=os.getenv("TIMEFRAME", raw.get("timeframe", "M5")),
        mt5=mt5_creds,
    )
