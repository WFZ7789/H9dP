"""Configuration, storage limits and atomic metadata writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

ROOT_LIMIT_BYTES = 375_000_000_000


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    config["config_path"] = str(path)
    config["repository_root"] = str(path.parent.parent)
    for name in ("manifest", "records", "labels", "cache_dir", "reuse_cache"):
        if name not in config["data"]:
            continue
        value = Path(config["data"][name])
        config["data"][name] = str(
            value.resolve() if value.is_absolute() else (path.parent.parent / value).resolve()
        )
    value = Path(config["output_dir"])
    config["output_dir"] = str(
        value.resolve() if value.is_absolute() else (path.parent.parent / value).resolve()
    )
    return config


def fingerprint(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_space(destination: str | Path, projected_bytes: int = 0) -> dict:
    """Enforce the repository's decimal-GB root limit before large writes."""
    destination = Path(destination).resolve()
    parent = destination
    while not parent.exists():
        parent = parent.parent
    root = shutil.disk_usage("/")
    target = shutil.disk_usage(parent)
    on_root = os.stat(parent).st_dev == os.stat("/").st_dev
    if root.used + (projected_bytes if on_root else 0) > ROOT_LIMIT_BYTES:
        raise RuntimeError("Root usage/projected usage exceeds 375,000,000,000 bytes")
    if target.free < projected_bytes + 1_000_000_000:
        raise RuntimeError(f"Insufficient storage at {destination}")
    return {
        "root_used_bytes": root.used,
        "root_limit_bytes": ROOT_LIMIT_BYTES,
        "destination_free_bytes": target.free,
        "projected_bytes": projected_bytes,
    }

