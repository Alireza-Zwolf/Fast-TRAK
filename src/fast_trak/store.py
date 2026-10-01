"""Guarding a TRAK store against being reused with a different configuration.

Features written under one model, prompt format or candidate file are
meaningless under another, and TRAK itself only checks the projection
dimension and the train-set size. The settings that determine the features
are therefore recorded next to them and compared on every later open.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def update_metadata(path: str | Path, expected: dict[str, Any], has_features: bool) -> None:
    """Record ``expected`` in the store metadata, or verify it is unchanged.

    Args:
        path: the store's ``metadata.json``.
        expected: settings that must stay fixed for the life of the store.
        has_features: whether the store already holds featurised checkpoints.
            A populated store that lacks one of the keys was written by other
            code and is rejected rather than adopted.

    Raises:
        ValueError: on any mismatch between stored and expected settings.
    """
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    mismatches = {
        key: {"stored": metadata[key], "requested": value}
        for key, value in expected.items()
        if key in metadata and metadata[key] != value
    }
    if mismatches:
        raise ValueError(
            f"{path.parent} was created with different settings: {mismatches}. "
            "Use a fresh save_dir."
        )
    missing = [key for key in expected if key not in metadata]
    if not missing:
        return
    if has_features:
        raise ValueError(
            f"{path.parent} already holds features but lacks {missing}; it was not "
            "created with these settings. Use a fresh save_dir."
        )

    metadata.update(expected)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
