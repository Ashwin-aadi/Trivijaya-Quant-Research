"""Factor construction, universe expansion, and book formation.

The first test is the one that matters: every signal must be computable from information available
strictly before the session it labels. The rest guard the two bugs actually hit while building this
module, both of which produced plausible output rather than an error.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.capacity.factors import (
    FACTOR_NAMES,
    build_factors,
    daily_universe,
    long_short_weights,
)
from src.common.exceptions import DataIntegrityError


def _panel(n: int = 400, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC")) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for s in symbols:
        closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, size=n))
        for i in range(n):
            rows.append(
                {
                    "session_date": date(2020, 1, 1) + timedelta(days=i),
                    "symbol": s,
                    "adj_close": float(closes[i]),
                    "turnover_inr": float(rng.lognormal(17, 0.4)),
                }
            )
    return pl.DataFrame(rows)


def test_no_factor_score_uses_the_session_it_labels() -> None:
    """Truncating the panel after session t must not change any score at or before t.

    This is the leakage test, and it is stronger than reading the code: it would catch a rolling
    window that forgot to shift, a signal built from a same-day close, or a normalisation computed
    over the whole sample. Any of those would let a factor see its own future and would look like
    an unusually good factor rather than like a bug.
    """
    panel = _panel(400)
    cut = date(2020, 1, 1) + timedelta(days=349)
    full = build_factors(panel).filter(pl.col("session_date") <= cut).sort(
        ["factor", "symbol", "session_date"]
    )
    truncated = build_factors(panel.filter(pl.col("session_date") <= cut)).sort(
        ["factor", "symbol", "session_date"]
    )
    assert full.height == truncated.height
    for column in ("factor", "symbol", "session_date"):
        assert full[column].to_list() == truncated[column].to_list()
    assert np.allclose(full["score"].to_numpy(), truncated["score"].to_numpy(), equal_nan=True)


def test_every_named_factor_is_actually_produced() -> None:
    """Guards the zoo silently shrinking, which would go unnoticed in an aggregate table."""
    scores = build_factors(_panel(400))
    assert set(scores["factor"].unique().to_list()) == set(FACTOR_NAMES)


def test_value_and_quality_are_absent_by_design() -> None:
    """They need fundamentals we do not hold; the absence is a stated limitation, not a slip."""
    assert "value" not in FACTOR_NAMES
    assert "quality" not in FACTOR_NAMES


def test_a_panel_missing_columns_raises() -> None:
    with pytest.raises(DataIntegrityError):
        build_factors(pl.DataFrame({"session_date": [date(2020, 1, 1)], "symbol": ["A"]}))


# --- universe expansion ---------------------------------------------------------------------


def test_daily_universe_expands_every_session_to_the_whole_membership() -> None:
    """The as-of join collapse: matching one right row per session gave 18 symbols, not 185.

    An as-of join returns a single match per left row. Joining the universe directly therefore
    yields one arbitrary symbol per session, which is a valid frame of the right shape carrying
    entirely wrong content — every downstream number was plausible and wrong.
    """
    universe = pl.DataFrame(
        {
            "rebalance_date": [date(2020, 1, 1)] * 3 + [date(2020, 4, 1)] * 3,
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "DDD"],
            "rank": [1, 2, 3, 1, 2, 3],
        }
    )
    sessions = [date(2020, 1, 1) + timedelta(days=i) for i in range(200)]
    expanded = daily_universe(universe, sessions)
    for day in (date(2020, 1, 15), date(2020, 6, 1)):
        members = expanded.filter(pl.col("session_date") == day)["symbol"].to_list()
        assert len(members) == 3, f"{day} expanded to {len(members)} symbols, expected 3"
    assert set(expanded.filter(pl.col("session_date") == date(2020, 6, 1))["symbol"]) == {
        "AAA", "BBB", "DDD"
    }


def test_sessions_before_the_first_rebalance_have_no_membership() -> None:
    """Borrowing the first rebalance backwards applies a later-selected universe to earlier days."""
    universe = pl.DataFrame(
        {"rebalance_date": [date(2020, 4, 1)] * 2, "symbol": ["AAA", "BBB"], "rank": [1, 2]}
    )
    sessions = [date(2020, 1, 1) + timedelta(days=i) for i in range(200)]
    expanded = daily_universe(universe, sessions)
    assert expanded.filter(pl.col("session_date") < date(2020, 4, 1)).height == 0


# --- book formation -------------------------------------------------------------------------


def test_the_book_is_rupee_neutral_and_both_legs_are_populated() -> None:
    scores = pl.DataFrame(
        {
            "session_date": [date(2020, 1, 1)] * 20,
            "factor": ["momentum_12_1"] * 20,
            "symbol": [f"S{i:02d}" for i in range(20)],
            "score": list(range(20)),
        }
    )
    book = long_short_weights(scores, quantile=0.2)
    assert book["weight"].sum() == pytest.approx(0.0, abs=1e-12)
    assert book.filter(pl.col("weight") > 0)["weight"].sum() == pytest.approx(1.0)
    assert book.filter(pl.col("weight") < 0)["weight"].sum() == pytest.approx(-1.0)
    # Top 4 of 20 long, bottom 4 short, at a 0.2 quantile.
    assert set(book.filter(pl.col("weight") > 0)["symbol"]) == {"S16", "S17", "S18", "S19"}


def test_a_cross_section_too_small_to_form_two_legs_is_dropped() -> None:
    scores = pl.DataFrame(
        {
            "session_date": [date(2020, 1, 1)] * 4,
            "factor": ["momentum_12_1"] * 4,
            "symbol": list("ABCD"),
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert long_short_weights(scores).height == 0


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.9, -0.1])
def test_an_impossible_quantile_raises(bad: float) -> None:
    scores = pl.DataFrame(
        {"session_date": [date(2020, 1, 1)], "factor": ["m"], "symbol": ["A"], "score": [1.0]}
    )
    with pytest.raises(DataIntegrityError):
        long_short_weights(scores, quantile=bad)
