from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Short-horizon mean reversion is based on the idea that assets which have "
        "deviated significantly from their historical price range are likely to revert "
        "to that mean. This strategy exploits the tendency of prices to revert after a "
        "sharp move, profiting from the convergence back to the mean."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_price = (history["close"].mean()).round(2).to_list()[0]
        symbols_above_mean = []
        for symbol in view.symbols:
            recent_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window + 1:
                continue
            last_close = recent_closes[-1]
            mean_of_recent_closes = sum(recent_closes[1:]) / (self._window)
            if last_close > mean_of_recent_closes * 1.25:  # Arbitrarily chosen threshold
                symbols_above_mean.append(symbol)

        if not symbols_above_mean:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_above_mean)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_above_mean}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest