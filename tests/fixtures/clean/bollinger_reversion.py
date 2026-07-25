"""Buy names trading below the lower Bollinger band."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, stdev


class BollingerReversion(Strategy):
    """Mean reversion against a volatility-scaled band."""

    rationale = (
        "Bollinger bands express distance from a moving average in units of the stock's own "
        "recent volatility, so a break below the lower band is a large move relative to how that "
        "name normally trades. The rule buys those, expecting reversion toward the average."
    )

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self._window = window
        self._num_std = num_std

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
            recent = values[-self._window:]
            average = sum(recent) / len(recent)
            dispersion = stdev(recent)
            if dispersion > 0 and values[-1] < average - self._num_std * dispersion:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
