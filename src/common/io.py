"""Disk I/O for raw and derived data, with the immutability guarantees the charter requires.

``data/raw/`` is write-once: every file is fetched from a source, hashed, and paired with a
``.meta.json`` sibling recording where it came from. Overwriting a raw file is treated as a bug,
not a convenience, so :func:`write_raw_parquet` refuses to clobber an existing one. Derived data
under ``interim/``/``processed/`` is regenerable and may be overwritten freely.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.common.exceptions import DataIntegrityError


def sha256_file(path: Path) -> str:
    """Streaming SHA256 of a file's bytes (chunked so large files don't load into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta_path(path: Path) -> Path:
    """The sibling metadata path for a raw data file: ``<name>.meta.json``."""
    return path.parent / (path.name + ".meta.json")


def write_raw_parquet(
    df: pl.DataFrame,
    path: Path,
    *,
    source_url: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a raw parquet file plus its ``.meta.json``, refusing to overwrite either.

    The metadata records source, fetch time, row count, and content hash so any raw file can be
    traced and integrity-checked later. Refusing to overwrite enforces the write-once rule
    structurally rather than by convention.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DataIntegrityError(f"raw file already exists, refusing to overwrite: {path}")
    df.write_parquet(path)

    meta = {
        "source_url": source_url,
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "row_count": df.height,
        "sha256": sha256_file(path),
        **(extra or {}),
    }
    meta_path(path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


def verify_raw(path: Path) -> dict[str, Any]:
    """Re-hash a raw file and confirm it matches its recorded metadata. Returns the metadata."""
    mp = meta_path(path)
    if not mp.exists():
        raise DataIntegrityError(f"raw file has no .meta.json sibling: {path}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if actual != meta.get("sha256"):
        raise DataIntegrityError(
            f"hash mismatch for {path}: recorded {meta.get('sha256')}, got {actual}"
        )
    return meta


def write_derived_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Write a regenerable parquet under interim/ or processed/. Overwriting is allowed here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a parquet file into a polars DataFrame."""
    if not path.exists():
        raise DataIntegrityError(f"parquet not found: {path}")
    return pl.read_parquet(path)
