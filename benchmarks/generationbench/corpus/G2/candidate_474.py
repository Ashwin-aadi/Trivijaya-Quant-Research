from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength compared to the broad market index tend to outperform "
        "over time. This is based on the idea that strong stocks are those whose performance "
        "outpaces the broader market, indicating a positive sentiment and potentially higher future returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = history.select(pl.col("adj_close")).to_numpy()
        benchmark_closes = view.closes(lookback=self._window).to_numpy()

        nifty100_returns = (nifty100_closes / nifty100_closes.shift(1) - 1.0)
        benchmark_returns = (benchmark_closes / benchmark_closes.shift(1) - 1.0)

        relative_strengths = [0] * len(view.symbols)
        for i, symbol in enumerate(view.symbols):
            if symbol not in nifty100_closes[:, 0]:
                continue
            stock_returns = nifty100_returns[i]
            benchmark_return = benchmark_returns[0]

            if (stock_returns - benchmark_return).mean() > 0:
                relative_strengths[i] += 1

        top_n_symbols = [symbol for _, symbol in sorted(zip(relative_strengths, view.symbols), reverse=True)[:5]]
        weights = {symbol: 1.0 / len(top_n_symbols) for symbol in top_n_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest