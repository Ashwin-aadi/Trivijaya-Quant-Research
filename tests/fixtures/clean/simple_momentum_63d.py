"""Hold the strongest performers over the past quarter."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n, window_return


class SimpleMomentum63d(Strategy):
    """Cross-sectional momentum over roughly one quarter of trading."""

    rationale = (
        "Medium-horizon momentum is among the most widely documented cross-sectional effects: "
        "names that have outperformed over the past few months have tended to keep doing so for "
        "a while. This holds the strongest decile and nothing else."
    )

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        scores = window_return(view, self._lookback)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
