"""Hold the names whose complete trailing window shows the least price dispersion.

Naming note: ``full_window`` is the trailing slice once it holds its full complement of sessions —
"full" describes the window's length, not the full sample. The slice is taken from
``view.closes``, which stops before the decision date, so a name with a short listing history is
skipped rather than scored on a partial window.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, top_n


class FullWindowVolatility(Strategy):
    """Ranks on the coefficient of variation of closes inside one complete trailing window."""

    rationale = (
        "Low-volatility stocks have historically delivered returns comparable to high-volatility "
        "ones with less risk, which is the low-volatility anomaly. Dividing the window's standard "
        "deviation by its mean makes the measure scale-free so names at different price levels "
        "compare. The weakness is that dispersion of prices is not the same as volatility of "
        "returns: a name in a steady trend has a wide price spread and calm daily moves, and this "
        "measure will call it volatile."
    )

    def __init__(self, window: int = 63, holdings: int = 10) -> None:
        self._window = window
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() < self._window:
                continue
            full_window = observed.tail(self._window)
            level = full_window.mean()
            spread = full_window.std()
            # A polars aggregate is typed as a union over every dtype the column might hold, so
            # narrow to float rather than asserting the dtype and hoping.
            if not isinstance(level, float) or not isinstance(spread, float) or level <= 0:
                continue
            scores[symbol] = spread / level
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
