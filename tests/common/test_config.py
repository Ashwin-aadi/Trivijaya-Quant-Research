"""Tests for the typed config loader: parsing, hashing, and the guardrails that reject a
malformed or self-contradictory configuration at load time rather than mid-run.
"""

import re
from datetime import date
from pathlib import Path

import pytest

from src.common.config import load_config
from src.common.exceptions import ConfigError


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_and_types(tmp_path: Path, good_config_text: str) -> None:
    cfg = load_config(write(tmp_path, good_config_text))
    assert cfg.meta.seed == 42
    assert cfg.dates.dev_start == date(2020, 1, 1)
    assert cfg.dates.holdout_start == date(2025, 1, 1)
    assert cfg.universe.size == 100
    assert cfg.universe.method == "liquidity_rank"
    # Buffer bands must straddle the target size or the rule churns / never drops anyone.
    assert cfg.universe.entry_rank <= cfg.universe.size <= cfg.universe.exit_rank
    assert cfg.data.prices.cross_check_rel_tol == pytest.approx(0.01)


def test_hash_is_content_stable(tmp_path: Path, good_config_text: str) -> None:
    # Same bytes at two paths -> same hash; the hash tracks content, not filename.
    p1 = tmp_path / "one.yaml"
    p1.write_text(good_config_text, encoding="utf-8")
    p2 = tmp_path / "two.yaml"
    p2.write_text(good_config_text, encoding="utf-8")
    assert load_config(p1).config_hash == load_config(p2).config_hash


def test_missing_section_rejected(tmp_path: Path, good_config_text: str) -> None:
    # Drop the whole `calendar:` block (its header plus its indented body) and confirm the loader
    # refuses rather than running with a hole in the configuration.
    lines = good_config_text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("calendar:"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" "))
    bad = "\n".join(lines[:start] + lines[end:])
    assert "calendar:" not in bad
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))


def test_holdout_must_follow_dev(tmp_path: Path, good_config_text: str) -> None:
    # holdout_start before dev_end would leak holdout data into development.
    bad = good_config_text.replace('holdout_start: "2025-01-01"', 'holdout_start: "2024-06-01"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))


def test_bad_date_rejected(tmp_path: Path, good_config_text: str) -> None:
    bad = re.sub(r'dev_start: "[^"]+"', 'dev_start: "not-a-date"', good_config_text)
    assert 'dev_start: "not-a-date"' in bad
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))
