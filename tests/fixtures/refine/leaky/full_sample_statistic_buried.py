"""Category ``full_sample_statistic``, variant ``buried``.

Defect: ``_scale`` divides each name's raw signal by a volatility figure computed once, in the
constructor, over every session in the supplied index series. The denominator is a whole-period
standard deviation, so the normalisation applied in 2016 already reflects the March 2020 crash. In
a calm stretch the scaled signal is systematically understated and in a violent one overstated,
in a way that happens to line up with what came next.

The file is a plausible volatility-targeted breakout book: a Donchian entry, an average-true-range
stop distance, a participation cap, and position sizing that targets a constant risk contribution.
Four of those five components are correct. The fifth is one division inside a two-line helper.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTargetedBreakout(Strategy):
    """Buys channel breakouts and sizes them to a constant risk contribution."""

    rationale = (
        "A breakout says the market has repriced a name, but the size of a breakout in rupees "
        "says nothing about how unusual it is. Dividing by a volatility estimate turns the raw "
        "move into a comparable quantity across names and across time, and sizing each position "
        "so that its expected risk contribution is equal stops one violent name from setting the "
        "return of the whole book."
    )

    def __init__(
        self,
        index_series: pl.Series,
        channel: int = 20,
        top_n: int = 10,
        risk_budget: float = 0.10,
    ) -> None:
        self._channel = channel
        self._top_n = top_n
        self._risk_budget = risk_budget
        # THE CHEAT: one standard deviation over the entire index series, computed before any
        # decision date exists. Every scaled signal the strategy ever produces is divided by a
        # number that summarises the whole period, the unseen part included.
        self._reference = max(_as_float(index_series.std()), 1e-6)

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._channel + 1)
        if closes.height < self._channel:
            return Signal(information_available_at=stamp, weights={})

        raw = self._breakout_strength(view, closes)
        if not raw:
            return Signal(information_available_at=stamp, weights={})

        scaled = {symbol: self._scale(value) for symbol, value in raw.items()}
        chosen = sorted(scaled, key=lambda s: (-scaled[s], s))[: self._top_n]
        return Signal(
            information_available_at=stamp,
            weights=self._size({symbol: scaled[symbol] for symbol in chosen}),
        )

    def _breakout_strength(self, view: MarketView, closes: pl.DataFrame) -> dict[str, float]:
        """How far each name has pushed above the top of its own channel, in rupees."""
        out: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) <= self._channel:
                continue
            ceiling = max(values[-self._channel - 1 : -1])
            if values[-1] > ceiling:
                out[symbol] = values[-1] - ceiling
        return out

    def _scale(self, value: float) -> float:
        """Express a raw breakout in units of the reference dispersion."""
        return value / self._reference

    def _size(self, scaled: dict[str, float]) -> dict[str, float]:
        """Spread the risk budget so each held name contributes the same expected risk."""
        if not scaled:
            return {}
        strength = {symbol: max(value, 1e-6) for symbol, value in scaled.items()}
        total = sum(strength.values())
        return {
            symbol: self._risk_budget * value / total for symbol, value in strength.items()
        }


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float; anything unusable becomes zero."""
    return float(value) if isinstance(value, int | float) else 0.0


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
