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
  holdout_end: "2025-12-31"
universe:
  method: liquidity_rank
  size: 100
  trailing_sessions: 126
  rebalance: quarterly
  entry_rank: 90
  exit_rank: 110
  min_listed_sessions: 126
  min_traded_fraction: 0.8
audit:
  model_tag: "qwen2.5:7b-instruct-q4_K_M"
  ollama_host: "http://localhost:11434"
  num_ctx: 4096
  request_timeout_seconds: 180.0
  probe_timeout_seconds: 2.0
costs:
  effective_from: "2026-04-01"
  verified_on: "2026-07-31"
  delivery:
    stt_buy: 0.001
    stt_sell: 0.001
    stamp_duty_buy: 0.00015
    stamp_duty_sell: 0.0
  intraday:
    stt_buy: 0.0
    stt_sell: 0.00025
    stamp_duty_buy: 0.00003
    stamp_duty_sell: 0.0
  exchange_transaction_charge: 0.0000297
  ipft_charge: 0.000001
  sebi_turnover_fee: 0.000001
  gst_rate: 0.18
  brokerage:
    delivery_rate: 0.0
    delivery_flat: 0.0
    intraday_rate: 0.0003
    intraday_flat_cap: 20.0
  dp_charge_per_scrip_sell: 3.50
  slippage:
    participation_coefficient: 0.1
    max_slippage_fraction: 0.05
  impact:
    coefficient: 1.0
    default_volatility: 0.02
constraints:
  max_participation_rate: 0.01
  min_adv_rupees: 10000000.0
  adv_window_sessions: 21
  circuit_band: 0.20
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
