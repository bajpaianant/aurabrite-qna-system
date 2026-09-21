"""Central configuration.

All runtime knobs are read once from environment variables (loaded from a
`.env` file if present) and then exposed as a frozen `Settings` instance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


LLMProvider = Literal["mock", "openai", "anthropic", "litellm"]
WebProvider = Literal["duckduckgo", "tavily", "none"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: LLMProvider = "mock"
    llm_model: str = "mock-supervisor"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Web search
    web_search_provider: WebProvider = "duckduckgo"
    tavily_api_key: str = ""

    # Paths
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    warehouse_path: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "warehouse" / "aurabrite.duckdb"
    )
    docs_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "documents")
    vector_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "warehouse" / "vector_index"
    )

    # Runtime
    max_agent_steps: int = 8
    validation_rounds: int = 2
    log_level: str = "INFO"


def load_settings() -> Settings:
    provider = _env("LLM_PROVIDER", "mock").lower()
    if provider not in ("mock", "openai", "anthropic", "litellm"):
        provider = "mock"

    web = _env("WEB_SEARCH_PROVIDER", "duckduckgo").lower()
    if web not in ("duckduckgo", "tavily", "none"):
        web = "none"

    settings = Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        llm_model=_env("LLM_MODEL", "mock-supervisor"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
        openai_api_key=_env("OPENAI_API_KEY", ""),
        anthropic_api_key=_env("ANTHROPIC_API_KEY", ""),
        web_search_provider=web,  # type: ignore[arg-type]
        tavily_api_key=_env("TAVILY_API_KEY", ""),
        data_dir=Path(_env("AURABRITE_DATA_DIR", str(REPO_ROOT / "data"))),
        warehouse_path=Path(
            _env(
                "AURABRITE_WAREHOUSE_PATH",
                str(REPO_ROOT / "data" / "warehouse" / "aurabrite.duckdb"),
            )
        ),
        docs_dir=Path(_env("AURABRITE_DOCS_DIR", str(REPO_ROOT / "data" / "documents"))),
        vector_dir=Path(
            _env(
                "AURABRITE_VECTOR_DIR",
                str(REPO_ROOT / "data" / "warehouse" / "vector_index"),
            )
        ),
        max_agent_steps=_env_int("AURABRITE_MAX_AGENT_STEPS", 8),
        validation_rounds=_env_int("AURABRITE_VALIDATION_ROUNDS", 2),
        log_level=_env("AURABRITE_LOG_LEVEL", "INFO").upper(),
    )
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    return settings


SETTINGS = load_settings()
