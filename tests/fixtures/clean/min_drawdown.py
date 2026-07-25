"""Hold the names that have fallen least from their recent peak."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class MinDrawdown(Strategy):
    """Ranks on distance below the recent high."""

    rationale = (
        "Drawdown from a recent peak is a different view of trend than a return: a name can post "
        "a positive quarter while still sitting well below its high. Selecting on shallow "
        "drawdowns favours steady advances over volatile paths that merely end higher."
    )

    def __init__(self, lookback: int = 126, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < 2:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < 2:
                continue
            peak = max(values)
            if peak <= 0:
                continue
            # A negative number; nearer zero means a shallower fall, so ranking largest-first
            # selects the least damaged names.
            scores[symbol] = values[-1] / peak - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
