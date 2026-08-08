from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion30d(Strategy):
    rationale = (
        "Mean reversion in stock prices suggests that stocks deviating significantly from their historical price ranges are likely to revert. By identifying such stocks and entering positions based on their z-scores, we aim to capture these mean-reverting opportunities while managing risk through strict entry and exit rules."
    )

    def __init__(self, threshold: float = 1.0, window: int = 30, max_positions: int = 30) -> None:
        self._threshold = threshold
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbols = [symbol for symbol in view.symbols if symbol in closes]

        mean_prices = {}
        std_devs = {}
        z_scores = {}

        for symbol in symbols:
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            mean_price = sum(prices[-self._window:]) / self._window
            std_dev = (sum([(p - mean_price) ** 2 for p in prices[-self._window:]]) / self._window) ** 0.5
            if std_dev == 0:
                continue

            z_score = (closes[symbol][-1] - mean_price) / std_dev
            mean_prices[symbol] = mean_price
            std_devs[symbol] = std_dev
            z_scores[symbol] = z_score

        sorted_symbols = [symbol for symbol, score in sorted(z_scores.items(), key=lambda item: abs(item[1]), reverse=True)]
        
        if len(sorted_symbols) < self._max_positions:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted_symbols[:self._max_positions]
        weight = 1.0 / len(top_symbols)
        entry_price = mean_prices[top_symbols[0]]
        stop_loss = entry_price * (1 - 0.1)  # 10% of the portfolio value

        signal_weights = {symbol: weight for symbol in top_symbols}
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols if z_scores[s] > self._threshold or mean_prices[s] < stop_loss}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest