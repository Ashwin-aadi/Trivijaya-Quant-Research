"""Tests for raw-data I/O: metadata is written, hashes verify, and the write-once rule holds."""

from pathlib import Path

import polars as pl
import pytest

from src.common.exceptions import DataIntegrityError
from src.common.io import meta_path, verify_raw, write_derived_parquet, write_raw_parquet


def sample_df() -> pl.DataFrame:
    return pl.DataFrame({"symbol": ["RELIANCE", "TCS"], "close": [2900.0, 3800.0]})


def test_raw_write_creates_meta(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    write_raw_parquet(sample_df(), path, source_url="https://example.test/bhavcopy")
    assert path.exists()
    meta = verify_raw(path)  # re-hashes and confirms it matches recorded metadata
    assert meta["row_count"] == 2
    assert meta["source_url"] == "https://example.test/bhavcopy"
    assert len(meta["sha256"]) == 64


def test_raw_write_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    write_raw_parquet(sample_df(), path, source_url="src")
    with pytest.raises(DataIntegrityError):
        write_raw_parquet(sample_df(), path, source_url="src")  # data/raw is immutable


def test_verify_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    write_raw_parquet(sample_df(), path, source_url="src")
    # Corrupt the metadata's recorded hash; verification must catch the mismatch.
    mp = meta_path(path)
    corrupted = mp.read_text(encoding="utf-8").replace('"sha256"', '"sha256_x"')
    mp.write_text(corrupted, encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        verify_raw(path)


def test_derived_write_allows_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "interim" / "features.parquet"
    write_derived_parquet(sample_df(), path)
    # Regenerable data may be rewritten without complaint.
    write_derived_parquet(sample_df(), path)
    assert path.exists()
