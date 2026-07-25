"""Hold names trading above their long-run moving average."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class PriceAboveSma200(Strategy):
    """A single long-horizon trend filter, with no ranking."""

    rationale = (
        "Price above its long-run average is the most widely used definition of a name being in "
        "an uptrend. This applies that filter and nothing else, so it isolates whether the "
        "filter alone carries information."
    )

    def __init__(self, window: int = 200) -> None:
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
            if values[-1] > sum(values[-self._window:]) / self._window:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
