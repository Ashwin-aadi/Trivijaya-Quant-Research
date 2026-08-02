"""Tests for the counterfactual panel reconstruction.

The load-bearing one is :func:`test_identity_path_reproduces_the_panel`. Resampling along the
identity sequence is a no-op, so the output must equal the input — and the first version of
:mod:`src.stress.panel` did not, because it measured returns over consecutive sessions instead of
consecutive available closes and so discarded each symbol's first quoted day. That was 48 rows of
211,927, which is invisible in any summary statistic and moved one strategy's Sharpe by 0.165.

The fixtures here are small and synthetic on purpose: a listing part-way through, a mid-series gap,
and a symbol quoted throughout, which are the three presence patterns the real panel contains.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.common.exceptions import DataIntegrityError
from src.stress.panel import SyntheticPanelBuilder

SESSIONS = 40


def _panel() -> pl.DataFrame:
    """Three symbols with different presence patterns, on a deterministic price path."""
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(SESSIONS)]
    rows: list[dict[str, object]] = []
    presence = {
        "ALWAYS": range(SESSIONS),
        "LISTED_LATE": range(12, SESSIONS),
        "HAS_A_GAP": [i for i in range(SESSIONS) if not 18 <= i <= 24],
    }
    for offset, (symbol, sessions) in enumerate(presence.items()):
        price = 100.0 * (1.0 + offset)
        for i in sessions:
            # A deterministic wiggle, so the series has structure without needing randomness.
            price *= 1.0 + 0.01 * np.sin(i + offset)
            rows.append({
                "session_date": days[i], "symbol": symbol,
                "open": price * 0.99, "high": price * 1.02, "low": price * 0.98,
                "close": price, "prev_close": price * 0.995, "volume": 1000.0 + i,
                "turnover_inr": price * (1000.0 + i), "divisor": 1.0,
                "adj_open": price * 0.99, "adj_high": price * 1.02, "adj_low": price * 0.98,
                "adj_close": price, "adj_volume": 1000.0 + i,
            })
    return pl.DataFrame(rows).sort("session_date")


def _universe(panel: pl.DataFrame) -> pl.DataFrame:
    symbols = sorted(panel["symbol"].unique().to_list())
    return pl.DataFrame({
        "rebalance_date": [date(2020, 1, 1)] * len(symbols),
        "symbol": symbols,
        "rank": list(range(1, len(symbols) + 1)),
    })


@pytest.fixture
def builder() -> SyntheticPanelBuilder:
    panel = _panel()
    return SyntheticPanelBuilder(panel, _universe(panel))


def test_identity_path_reproduces_the_panel(builder: SyntheticPanelBuilder) -> None:
    """The whole contract in one assertion: a no-op resampling must change nothing."""
    panel = _panel()
    synthetic, _ = builder.build(np.arange(SESSIONS - 1))

    left = panel.sort(["session_date", "symbol"])
    right = synthetic.sort(["session_date", "symbol"])
    assert right.height == left.height
    assert right.columns == left.columns
    for column in left.columns:
        if left.schema[column] != pl.Float64:
            assert right[column].to_list() == left[column].to_list()
            continue
        # Compounding 39 log steps costs a few ulps; anything structural is far larger.
        assert right[column].to_numpy() == pytest.approx(left[column].to_numpy(), rel=1e-12)


def test_identity_path_keeps_every_row_including_first_quotations(
    builder: SyntheticPanelBuilder,
) -> None:
    """The specific regression: a symbol's first quoted day must survive.

    ``LISTED_LATE`` first trades at session 12 and ``HAS_A_GAP`` resumes at session 25. Both were
    dropped by the session-differencing version, along with every real IPO in the corpus.
    """
    panel = _panel()
    synthetic, diagnostics = builder.build(np.arange(SESSIONS - 1))

    present = set(
        zip(synthetic["session_date"].to_list(), synthetic["symbol"].to_list(), strict=True)
    )
    expected = set(zip(panel["session_date"].to_list(), panel["symbol"].to_list(), strict=True))
    assert present == expected
    # One carried-forward row per symbol: its first ever quotation, which has no prior close.
    assert diagnostics.carried_forward_days == 3


def test_a_resampled_path_never_splices_price_levels(builder: SyntheticPanelBuilder) -> None:
    """Levels are compounded, so no synthetic move may exceed the largest real one.

    This is the property that fails loudly if the resampling is ever applied to price rows: the
    real panel spans a 2.55x median level drift, so level-splicing shows up here as a move of
    several hundred percent rather than of a few.
    """
    panel = _panel()
    largest = (
        panel.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1).alias("r")
        )["r"].abs().max()
    )
    real = float(largest)  # type: ignore[arg-type]
    rng = np.random.default_rng(42)
    for _ in range(20):
        path = rng.integers(0, SESSIONS - 1, size=SESSIONS - 1)
        _, diagnostics = builder.build(path)
        assert diagnostics.max_abs_session_return <= real + 1e-12


def test_volume_and_traded_value_come_from_the_sampled_day(
    builder: SyntheticPanelBuilder,
) -> None:
    """Liquidity travels with its date unchanged; only prices are reconstructed."""
    panel = _panel()
    sessions = sorted(panel["session_date"].unique().to_list())
    # Hold every synthetic session at one sampled day, so the expected volume is unambiguous.
    sampled = 7
    synthetic, _ = builder.build(np.full(SESSIONS - 1, sampled - 1))

    source = panel.filter(pl.col("session_date") == sessions[sampled]).sort("symbol")
    got = synthetic.filter(pl.col("session_date") == sessions[5]).sort("symbol")
    assert got["volume"].to_list() == source["volume"].to_list()
    assert got["turnover_inr"].to_list() == source["turnover_inr"].to_list()


def test_the_output_is_session_sorted(builder: SyntheticPanelBuilder) -> None:
    """PanelIndex refuses an unsorted panel, and the engine builds one from this directly."""
    rng = np.random.default_rng(7)
    synthetic, _ = builder.build(rng.integers(0, SESSIONS - 1, size=SESSIONS - 1))
    assert synthetic["session_date"].is_sorted()


def test_the_output_carries_the_sorted_flag(builder: SyntheticPanelBuilder) -> None:
    """Sortedness metadata must match the real panel's, not merely the row order.

    Polars treats a sorted-and-flagged column differently from a sorted-but-unflagged one: a
    ``.sort(descending=True)`` reverses the first and genuinely re-sorts the second, and the two
    disagree on tied rows. ``candidate_072`` does exactly that, and the mismatch handed it disjoint
    portfolios — the alphabetically last five symbols against the alphabetically first five — from
    inputs agreeing to 7e-16. Values being equal is not sufficient; the metadata is observable.
    """
    panel = _panel()
    synthetic, _ = builder.build(np.arange(SESSIONS - 1))
    assert synthetic["session_date"].flags == panel.sort("session_date")["session_date"].flags


def test_a_path_of_the_wrong_length_is_refused(builder: SyntheticPanelBuilder) -> None:
    with pytest.raises(DataIntegrityError, match="steps"):
        builder.build(np.arange(SESSIONS))


def test_the_missing_member_rate_is_measured_against_the_real_one(
    builder: SyntheticPanelBuilder,
) -> None:
    """Option (a) alignment costs some member-days; the diagnostic must expose the cost.

    On an identity path the two rates are equal by construction — a synthetic rate that did not
    collapse onto the real one there would mean the diagnostic was measuring something else.
    """
    _, identity = builder.build(np.arange(SESSIONS - 1))
    assert identity.missing_member_rate == pytest.approx(identity.real_missing_member_rate)

    rng = np.random.default_rng(3)
    _, resampled = builder.build(rng.integers(0, SESSIONS - 1, size=SESSIONS - 1))
    assert resampled.missing_member_rate >= 0.0
    assert resampled.universe_member_days == SESSIONS * 3
