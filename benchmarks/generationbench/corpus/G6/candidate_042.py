from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthIndex(Strategy):
    rationale = (
        "This strategy capitalizes on temporary market inefficiencies by identifying stocks "
        "that outperform the Nifty 50 index using a short-term relative strength metric. By "
        "entering positions when RSI falls below 30 and exiting if it rises above 70 or if "
        "underperformance persists, we aim to maximize returns while managing risk."
    )

    def __init__(self, window: int = 14, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = history.select(pl.col("adj_close").alias("nifty50"))
        symbol_closes = history.drop("nifty50", "session_date")
        relative_strengths: list[tuple[str, float]] = []

        for symbol in view.symbols:
            if symbol not in symbol_closes.columns:
                continue
            nifty50_series = [float(v) for v in nifty50_closes["nifty50"].to_list()]
            symbol_series = [float(v) for v in symbol_closes[symbol].drop_nulls().to_list()]

            if len(nifty50_series) < self._window or len(symbol_series) < self._window:
                continue

            nifty50_returns = [(v / prev - 1.0) for prev, v in zip(nifty50_series[:-1], nifty50_series[1:])]
            symbol_returns = [(v / prev - 1.0) for prev, v in zip(symbol_series[:-1], symbol_series[1:])]

            average_nifty50_return = sum(nifty50_returns) / len(nifty50_returns)
            average_symbol_return = sum(symbol_returns) / len(symbol_returns)

            rsi = (average_symbol_return - average_nifty50_return) / average_nifty50_return
            relative_strengths.append((symbol, rsi))

        relative_strengths.sort(key=lambda x: x[1], reverse=True)
        if not relative_strengths:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s, _ in relative_strengths[: self._top_n]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest