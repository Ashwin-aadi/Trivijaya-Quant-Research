from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Historical evidence suggests that low-volatility stocks tend to outperform high-volatility "
        "stocks over long periods. By constructing a portfolio focused on low-volatility names, we aim "
        "to capture higher average returns while reducing exposure to extreme market movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        stock_volatility = {}
        symbols = [sym for sym in view.symbols if sym in history["symbol"].to_list()]
        for symbol in symbols:
            daily_returns = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("adj_close").drop_nulls().shift(-1) / pl.col("adj_close").drop_nulls() - 1.0
                )
                .sort("session_date")
                .collect()["close"]
            )
            if daily_returns.height < self._window:
                continue
            volatility = daily_returns.std()
            stock_volatility[symbol] = volatility

        ranked_symbols = sorted(stock_volatility.keys(), key=lambda x: stock_volatility[x])
        bottom_quartile_count = int(len(symbols) / 4)
        selected_symbols = ranked_symbols[:bottom_quartile_count]
        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest