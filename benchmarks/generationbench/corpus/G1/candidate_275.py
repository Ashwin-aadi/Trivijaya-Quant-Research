from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy combines two characteristics: a 20-day closing price increase and "
        "a relative strength index (RSI) reading below 30. These conditions suggest overbought "
        "conditions have eased and the stock may be poised for a rebound."
    )

    def __init__(self, rsi_window: int = 14, breakout_window: int = 20) -> None:
        self._rsi_window = rsi_window
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window)

        if history.height < self._breakout_window:
            return Signal(information_available_at=stamp, weights={})

        rsi_history = view.history(lookback=self._rsi_window)

        if rsi_history.height < self._rsi_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            last_adj_close = adj_closes[-1]
            second_last_adj_close = adj_closes[-2]

            if (last_adj_close > second_last_adj_close and
                    all(adj_close >= second_last_adj_close for adj_close in adj_closes[:-2])):
                breakout_symbols.append(symbol)

        rsi_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in rsi_history.columns:
                continue
            rsi_values = [float(v) for v in rsi_history[symbol]["adj_close"].drop_nulls().to_list()]
            last_rsi_value = rsi_values[-1]

            if last_rsi_value < 30.0:
                rsi_symbols.append(symbol)

        combined_symbols = list(set(breakout_symbols).intersection(rsi_symbols))
        if not combined_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in combined_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest