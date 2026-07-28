"""Sort the universe into a high and a low trailing-return bucket and hold the high one.

Naming note: each symbol is given a ``label`` of ``"high"`` or ``"low"``. The word names a
readable bucket for whoever reads the position list; it is not a supervised-learning target. The
buckets are cut from returns that had already occurred by the last visible session, so nothing
here is the answer to a prediction problem.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, window_return


class LabelledMomentumBuckets(Strategy):
    """Splits the universe at its median trailing return and holds the upper half."""

    rationale = (
        "Cross-sectional momentum is the observation that names which have risen relative to "
        "their peers tend to keep doing so over the following months. Splitting at the median "
        "rather than taking a fixed count keeps the portfolio's breadth stable as the universe "
        "changes size. Holding half the universe makes this a diluted version of the effect: it "
        "will track the index closely and is unlikely to clear costs."
    )

    def __init__(self, lookback: int = 126) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        # Symbol breaks ties so the cut is reproducible when two names post identical returns.
        ordered = sorted(returns.items(), key=lambda item: (-item[1], item[0]))
        cut = max(1, len(ordered) // 2)

        buckets: dict[str, str] = {}
        for position, (symbol, _) in enumerate(ordered):
            label = "high" if position < cut else "low"
            buckets[symbol] = label

        picks = sorted(symbol for symbol, bucket in buckets.items() if bucket == "high")
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
