"""Hold names whose latest visible close sits below a low quantile of their trailing window."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class TrailingQuantileFilter(Strategy):
    """Compares the last close against a quantile of the same name's trailing closes."""

    rationale = (
        "A name near the bottom of its own recent price range has fallen without, so far, any "
        "sign of stabilising, and mean reversion over horizons of a few weeks is a documented if "
        "modest effect in equities. Using a quantile rather than the minimum makes the measure "
        "insensitive to one bad print. The obvious failure mode is that a name can be at the "
        "bottom of its range because something is genuinely wrong with it, which this cannot see."
    )

    def __init__(self, window: int = 63, quantile: float = 0.2) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("the quantile must lie in (0, 1)")
        self._window = window
        self._quantile = quantile

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() < self._window:
                continue
            # The quantile is taken over the trailing slice alone. Taken over the whole series it
            # would rank today's close against prices that had not been set at decision time,
            # which is the version of this construction that does leak.
            trailing = observed.tail(self._window)
            cutoff = trailing.quantile(self._quantile)
            if cutoff is None:
                continue
            if float(observed[-1]) <= cutoff:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
