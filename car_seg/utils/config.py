
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dict subclass with attribute-style access. Nested dicts are auto-wrapped."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v: Any) -> Any:
        if isinstance(v, dict):
            return Config(v)
        if isinstance(v, list):
            return [Config._wrap(x) for x in v]
        return v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401
        return self[key] if key in self else default

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def merge_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply CLI-style key=value overrides (dot-paths supported).

    Example: ["train.lr=1e-4", "model.num_classes=10"]
    """
    out = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, raw_val = ov.split("=", 1)
        # YAML-parse the value so "true"/"3"/"1e-4" become correct types
        val = yaml.safe_load(raw_val)
        node: Any = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = Config._wrap(val)
    return out
