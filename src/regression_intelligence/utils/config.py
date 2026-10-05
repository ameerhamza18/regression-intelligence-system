from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from regression_intelligence.utils.paths import CONFIG_DIR


class ProjectConfig(BaseModel):
    name: str
    random_seed: int = 42


class DataConfig(BaseModel):
    train_filename: str
    store_filename: str
    target: str
    date_column: str


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    val_size: float = Field(gt=0, lt=1)


class TrainingConfig(BaseModel):
    cv_folds: int = Field(ge=2)
    scoring: str


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    training: TrainingConfig
    logging: LoggingConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str | None = None) -> Config:
    env = env or os.getenv("APP_ENV", "dev")
    merged = _read_yaml(CONFIG_DIR / "base.yaml")
    env_file = CONFIG_DIR / f"{env}.yaml"

    if not env_file.exists():
        raise FileNotFoundError(f"No config for environment '{env}': {env_file}")
    merged = _deep_merge(merged, _read_yaml(env_file))
    return Config(**merged)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
