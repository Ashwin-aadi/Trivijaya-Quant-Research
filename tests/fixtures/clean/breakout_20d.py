"""Hold names closing at a twenty-session high."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class Breakout20d(Strategy):
    """Buys new highs over a one-month window."""

    rationale = (
        "A close at the top of its recent range means every holder from that period is in "
        "profit, so the supply of sellers waiting to break even is exhausted. That is the "
        "standard argument for entering on a breakout."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._window:
                continue
            if values[-1] >= max(values[-self._window:]):
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
