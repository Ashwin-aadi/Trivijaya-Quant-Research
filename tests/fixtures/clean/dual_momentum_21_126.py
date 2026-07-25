"""Require momentum to be positive over both a short and a long window."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, window_return


class DualMomentum21x126(Strategy):
    """Holds only names trending up on both horizons."""

    rationale = (
        "Demanding agreement between a one-month and a six-month window filters out names whose "
        "recent strength contradicts their longer trend, which is a common way to reduce the "
        "whipsaw a single-window momentum rule suffers."
    )

    def __init__(self, short_lookback: int = 21, long_lookback: int = 126) -> None:
        self._short = short_lookback
        self._long = long_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        short_returns = window_return(view, self._short)
        long_returns = window_return(view, self._long)
        picks = sorted(
            symbol for symbol in short_returns
            if short_returns[symbol] > 0 and long_returns.get(symbol, -1.0) > 0
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
