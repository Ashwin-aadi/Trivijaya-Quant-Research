"""Category ``point_in_time_bypass``, variant ``buried``.

Defect: the data does not arrive through the constructor at all. A module-level dictionary holds
the price frames, populated once at set-up time by ``register_reference``, and ``_make_lookup``
builds a closure over it. The closure is stored on the instance and called inside ``generate``, so
the decision function reads a frame that no argument ever passed to it and that nothing truncated.

This is the hardest form of the category to see, because the constructor is clean. It takes four
integers. Every argument to ``generate`` is legitimate. The bulk data enters through import-time
module state and leaves through a function object, and neither of those is visible at either
boundary a reviewer would think to check.

The surrounding strategy is a plausible pairs-style dispersion book: a correlation screen, a
spread z-score, an entry band, and a hard cap on the number of legs.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

# Module-level store, filled once during set-up and read for the life of the process. Nothing here
# is truncated to a decision date, because nothing here knows what a decision date is.
_REFERENCE: dict[str, pl.DataFrame] = {}


def register_reference(name: str, rows: pl.DataFrame) -> None:
    """Record a frame under ``name`` so strategies in this module can reach it later."""
    _REFERENCE[name] = rows


def _make_lookup(name: str) -> Callable[[str], list[float]]:
    """Build a reader over the registered frame ``name``, keyed by symbol."""

    def read(symbol: str) -> list[float]:
        # THE CHEAT: the closure holds `_REFERENCE`, so calling it from inside `generate` returns
        # the complete series for a symbol, every session in the sample included. The point-in-time
        # view never sees this call and cannot restrict it.
        frame = _REFERENCE.get(name)
        if frame is None:
            return []
        return [float(v) for v in frame.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]

    return read


class DispersionPairsBook(Strategy):
    """Holds the cheap leg of each pair whose spread has stretched past its usual band."""

    rationale = (
        "Two names driven by the same underlying exposure should not drift apart indefinitely, so "
        "the spread between them is a more stationary object than either price. Trading the "
        "spread when it stretches beyond its usual band, rather than trading either name "
        "outright, removes the common exposure and leaves the relative move, which is the part "
        "the signal actually claims to know something about."
    )

    def __init__(
        self,
        window: int = 63,
        entry_band: float = 2.0,
        max_legs: int = 8,
        min_overlap: int = 40,
    ) -> None:
        self._window = window
        self._entry_band = entry_band
        self._max_legs = max_legs
        self._min_overlap = min_overlap
        self._lookup = _make_lookup("prices")

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        candidates = [s for s in view.symbols if s in closes.columns]
        if len(candidates) < 2:
            return Signal(information_available_at=stamp, weights={})

        stretched = self._stretched_legs(candidates)
        chosen = sorted(stretched, key=lambda s: (stretched[s], s))[: self._max_legs]
        return Signal(information_available_at=stamp, weights=_spread(chosen))

    def _stretched_legs(self, candidates: list[str]) -> dict[str, float]:
        """Spread z-score per name against the equal-weighted basket of its peers."""
        out: dict[str, float] = {}
        basket = self._basket(candidates)
        if len(basket) < self._min_overlap:
            return out
        for symbol in candidates:
            series = self._lookup(symbol)
            if len(series) < self._min_overlap:
                continue
            overlap = min(len(series), len(basket))
            spread = [series[i] - basket[i] for i in range(overlap)]
            score = _zscore(spread)
            if score < -self._entry_band:
                out[symbol] = score
        return out

    def _basket(self, candidates: list[str]) -> list[float]:
        """Equal-weighted average level across the candidate names, session by session."""
        series = [self._lookup(symbol) for symbol in candidates]
        usable = [s for s in series if len(s) >= self._min_overlap]
        if not usable:
            return []
        overlap = min(len(s) for s in usable)
        return [sum(s[i] for s in usable) / len(usable) for i in range(overlap)]


def _zscore(values: list[float]) -> float:
    """Standardised distance of the last observation from the mean of ``values``."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    spread = float(variance**0.5)
    if spread <= 0:
        return 0.0
    return (values[-1] - mean) / spread


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
