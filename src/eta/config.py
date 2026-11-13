"""Load config.yaml and resolve paths against the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config into a plain dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path, root: Path = ROOT) -> Path:
    """Turn a repo-relative path from the config into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else root / p
