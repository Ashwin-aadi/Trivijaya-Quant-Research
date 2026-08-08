from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trends in stock prices by scaling them with their historical "
        "volatility. High volatility suggests uncertainty and potential reversals, while low "
        "volatility indicates sustained trends. The approach combines elements of trend-following "
        "and risk management to ensure dynamic position sizing and timely exits."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        closes = history["close"][symbols].to_dict()
        open_prices = history["open"][symbols].to_dict()
        high_prices = history["high"][symbols].to_dict()
        low_prices = history["low"][symbols].to_dict()
        volume = history["volume"][symbols].to_dict()

        returns = {
            symbol: [float(closes[symbol][i + 1] / closes[symbol][i] - 1)
                     for i in range(len(closes[symbol]) - 1)] 
            for symbol in symbols
        }
        sma_20 = {symbol: sum(returns[symbol][:self._window]) / self._window for symbol in symbols}
        rsv_20 = {
            symbol: (float(high_prices[symbol][-1] - low_prices[symbol][-1])) /
                    ((high_prices[symbol][-1] - low_prices[symbol][-1]) + 2 * abs(sma_20[symbol] - closes[symbol][-1]))
            for symbol in symbols
        }

        breakout_symbols = []
        for symbol in symbols:
            if (
                len(returns[symbol]) >= self._window and 
                returns[symbol][-1] > sma_20[symbol] + 2 * rsv_20[symbol]
            ):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest