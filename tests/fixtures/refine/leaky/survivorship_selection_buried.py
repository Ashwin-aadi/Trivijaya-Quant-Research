"""Category ``survivorship_selection``, variant ``buried``.

Defect: ``_eligible`` intersects the point-in-time universe with a membership set built from the
newest snapshot in the constituent frame, regardless of the decision date. The strategy therefore
only ever considers names that were still in the index at the end of the sample.

The intersection is two lines inside a helper that also does three legitimate things — a liquidity
floor, a warm-up requirement, and a per-sector cap — and the file around it implements a plausible
sector-neutral momentum book with turnover control. The visible behaviour of the file is a
reasonable strategy; the removal of everything that later failed happens in passing.

This is the most damaging category precisely because it leaves no mark in the trading logic. A
reviewer checking the signal computation line by line will find nothing wrong with it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class SectorNeutralMomentum(Strategy):
    """Momentum book with a per-sector cap and a liquidity floor."""

    rationale = (
        "Unconstrained cross-sectional momentum in Indian large caps collapses into whichever "
        "sector is currently in favour, so the book ends up as a single macro bet wearing a "
        "quantitative label. Capping the number of names per sector keeps the position count "
        "spread across the market while preserving the ranking, and a liquidity floor keeps the "
        "book out of names where the slippage model says the edge would be spent on execution."
    )

    def __init__(
        self,
        constituents: pl.DataFrame,
        sectors: dict[str, str],
        lookback: int = 126,
        top_n: int = 15,
        per_sector: int = 3,
    ) -> None:
        self._constituents = constituents
        self._sectors = sectors
        self._lookback = lookback
        self._top_n = top_n
        self._per_sector = per_sector
        # THE CHEAT: the newest snapshot in the constituent frame is resolved once, here, and used
        # for every decision date the strategy will ever face. Membership as at the end of the
        # sample is not membership as at the decision date; every name delisted or demoted along
        # the way is removed from consideration on days when it was tradable and falling.
        newest = constituents["snapshot_date"].max()
        self._members: frozenset[str] = frozenset(
            constituents.filter(pl.col("snapshot_date") == newest)["symbol"].to_list()
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        scores = self._momentum(view, closes)
        if not scores:
            return Signal(information_available_at=stamp, weights={})

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        picks = self._apply_sector_cap([symbol for symbol, _ in ordered])
        return Signal(information_available_at=stamp, weights=_spread(picks))

    def _momentum(self, view: MarketView, closes: pl.DataFrame) -> dict[str, float]:
        """Trailing return per eligible name, skipping the most recent month.

        Dropping the last twenty-one sessions is the standard construction: the very recent leg of
        a momentum window is dominated by short-term reversal, which pulls in the opposite
        direction to the effect the rest of the window is measuring.
        """
        out: dict[str, float] = {}
        for symbol in self._eligible(view, closes):
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback or values[0] <= 0:
                continue
            end = values[-22] if len(values) > 22 else values[-1]
            out[symbol] = end / values[0] - 1.0
        return out

    def _eligible(self, view: MarketView, closes: pl.DataFrame) -> list[str]:
        """Names with enough history, a known sector, and a place on the membership list."""
        out: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            if symbol not in self._sectors:
                continue
            if closes[symbol].drop_nulls().len() < self._lookback:
                continue
            if symbol not in self._members:
                continue
            out.append(symbol)
        return out

    def _apply_sector_cap(self, ordered: list[str]) -> list[str]:
        """Walk the ranking in order, taking names until each sector's quota is used up."""
        taken: dict[str, int] = defaultdict(int)
        picks: list[str] = []
        for symbol in ordered:
            sector = self._sectors.get(symbol, "unknown")
            if taken[sector] >= self._per_sector:
                continue
            taken[sector] += 1
            picks.append(symbol)
            if len(picks) >= self._top_n:
                break
        return picks


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
