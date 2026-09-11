"""Turn a ``config/specdec/*.yaml`` into the ``--speculative-config`` server argument (W4, ADR 0015).

The drivers pass extra server flags through an unquoted shell variable, so the JSON is rendered
without any whitespace: one shell word, exactly what W1 passed by hand when it loaded the
EAGLE-3 head. Keys are sorted so the same YAML always yields the same string in the manifests.
``none.yaml`` (``speculative_config: null``) renders to an empty string.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_specdec(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def speculative_config_json(cfg: dict[str, Any]) -> str | None:
    spec = cfg.get("speculative_config")
    if not spec:
        return None
    return json.dumps(spec, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def speculative_config_flag(path: Path) -> str:
    payload = speculative_config_json(load_specdec(path))
    return "" if payload is None else f"--speculative-config {payload}"
