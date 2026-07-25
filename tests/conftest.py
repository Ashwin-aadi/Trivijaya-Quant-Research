"""Shared test fixtures.

A single valid config document lives here so tests don't import each other. Individual tests that
probe parser edge cases mutate ``good_config_text`` and write their own file.
"""

from pathlib import Path

import pytest

from src.common.config import Config, load_config

GOOD_CONFIG_TEXT = """
meta:
  seed: 42
  project: test-lab
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  runs: runs
calendar:
  exchange: NSE
  history_start: "2019-01-01"
  index_symbol: "^CNX100"
dates:
  dev_start: "2020-01-01"
  dev_end: "2024-12-31"
  holdout_start: "2025-01-01"
universe:
  method: liquidity_rank
  size: 100
  trailing_sessions: 126
  rebalance: quarterly
  entry_rank: 90
  exit_rank: 110
  min_listed_sessions: 126
  min_traded_fraction: 0.8
data:
  prices:
    authoritative_source: bhavcopy
    cross_check_rel_tol: 0.01
    max_discrepancy_rate: 0.03
"""


@pytest.fixture
def good_config_text() -> str:
    return GOOD_CONFIG_TEXT


@pytest.fixture
def config_with_runs_in_tmp(tmp_path: Path) -> Config:
    """A loaded config whose runs/ directory points inside the test's tmp dir."""
    text = GOOD_CONFIG_TEXT.replace("runs: runs", f"runs: {tmp_path.as_posix()}/runs")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return load_config(cfg_path)
