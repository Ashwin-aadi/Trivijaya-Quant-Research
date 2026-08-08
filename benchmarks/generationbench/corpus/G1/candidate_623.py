from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "Combining a 20-day moving average crossover with relative strength provides a balanced "
        "approach to selecting stocks. The moving average gives a trend signal while relative "
        "strength indicates which assets are performing better than the market."
    )

    def __init__(self, ma_window: int = 20, rs_window: int = 10) -> None:
        self._ma_window = ma_window
        self._rs_window = rs_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._ma_window + self._rs_window)
        if closes.height < self._ma_window + self._rs_window:
            return Signal(information_available_at=stamp, weights={})

        ma_values = []
        rs_values = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._ma_window + self._rs_window)
            close_prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            ma_close_price = sum(close_prices[-self._ma_window:]) / self._ma_window
            ma_values.append(ma_close_price)

            rs_history = history.select(
                pl.col("symbol"), pl.col("adj_close").sort(descending=True).head(self._rs_window)
            ).collect()
            rs_close_prices = [float(v) for v in rs_history[symbol].to_list()]
            if len(rs_close_prices) >= self._rs_window:
                avg_rs_close_price = sum(rs_close_prices[-self._rs_window:]) / self._rs_window
                rs_values.append(avg_rs_close_price)
            else:
                rs_values.append(None)

        ma_ranked = sorted(zip(view.symbols, ma_values), key=lambda x: -x[1])
        rs_ranked = sorted(zip(view.symbols, rs_values), key=lambda x: -x[1])

        selected_symbols = set()
        for symbol, _ in ma_ranked[:5]:
            if rs_values[symbols.index(symbol)] is not None and \
                    (symbol not in selected_symbols):
                selected_symbols.add(symbol)

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest