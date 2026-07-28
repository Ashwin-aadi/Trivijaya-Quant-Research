"""Hold names trading above their own trailing average, keeping a fixed share in cash.

Naming note: the dictionary handed to the signal is called ``final_weights`` because it is the
last of two weighting steps inside ``generate``. ``final`` describes a stage of this function, not
end-of-period index membership — the universe comes from ``view.symbols``, which is point-in-time
and still contains names that were later removed.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class FinalWeightsAssembly(Strategy):
    """A trend filter whose weights are built in two passes, the second holding back cash."""

    rationale = (
        "A price above its own trailing average is the plainest available definition of an "
        "uptrend, and this holds the names in that state equally. The cash reserve is a blunt "
        "risk control rather than a signal: it lowers gross exposure uniformly, which reduces "
        "both the return and the drawdown roughly in proportion, and it is here mainly so that "
        "the weight assembly has more than one step."
    )

    def __init__(self, window: int = 100, cash_reserve: float = 0.1) -> None:
        if not 0.0 <= cash_reserve < 1.0:
            raise ValueError("the cash reserve must lie in [0, 1)")
        self._window = window
        self._cash_reserve = cash_reserve

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            average = sum(values) / len(values)
            if values[-1] > average:
                picks.append(symbol)

        # First pass spreads capital evenly over the picks; the second scales every weight down by
        # the reserve. The dictionary that comes out of the second pass is what the engine trades.
        even_weights = equal_weight(sorted(picks))
        final_weights = {s: w * (1.0 - self._cash_reserve) for s, w in even_weights.items()}
        return Signal(information_available_at=stamp, weights=final_weights)
