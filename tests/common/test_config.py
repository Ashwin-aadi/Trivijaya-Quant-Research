"""Tests for the typed config loader: parsing, hashing, and the guardrails that reject a
malformed or self-contradictory configuration at load time rather than mid-run.
"""

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
    assert cfg.dates.dev_start == date(2015, 1, 1)
    assert cfg.dates.holdout_start == date(2024, 1, 1)
    assert cfg.universe.membership_source is None
    assert cfg.data.prices.cross_check_rel_tol == pytest.approx(0.01)


def test_hash_is_content_stable(tmp_path: Path, good_config_text: str) -> None:
    # Same bytes at two paths -> same hash; the hash tracks content, not filename.
    p1 = tmp_path / "one.yaml"
    p1.write_text(good_config_text, encoding="utf-8")
    p2 = tmp_path / "two.yaml"
    p2.write_text(good_config_text, encoding="utf-8")
    assert load_config(p1).config_hash == load_config(p2).config_hash


def test_missing_section_rejected(tmp_path: Path, good_config_text: str) -> None:
    bad = good_config_text.replace(
        'calendar:\n  exchange: NSE\n  history_start: "2015-01-01"\n', ""
    )
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))


def test_holdout_must_follow_dev(tmp_path: Path, good_config_text: str) -> None:
    # holdout_start before dev_end would leak holdout data into development.
    bad = good_config_text.replace('holdout_start: "2024-01-01"', 'holdout_start: "2023-06-01"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))


def test_bad_date_rejected(tmp_path: Path, good_config_text: str) -> None:
    bad = good_config_text.replace('dev_start: "2015-01-01"', 'dev_start: "not-a-date"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad))
