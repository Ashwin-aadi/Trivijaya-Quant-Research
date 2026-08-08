from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks with strong relative performance can provide a "
        "positive return profile. This strategy selects the top N performing "
        "stocks based on their returns against the broad market index."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_returns = history.select(
            pl.col("adj_close").shift(-1).alias("prev_adj_close")
        ).with_columns(
            (pl.col("adj_close") / pl.col("prev_adj_close") - 1.0).alias("return")
        )

        stock_returns = view.closes(lookback=self._window)
        for symbol in view.symbols:
            if symbol not in stock_returns.columns:
                continue
            stock_returns[symbol] = [float(v) for v in stock_returns[symbol].to_list()]
        
        market_avg_return = market_returns.select(pl.col("return").mean()).item()
        top_stocks: list[str] = []
        for symbol, returns in stock_returns.to_dict().items():
            if len(returns) < self._window:
                continue
            avg_return = sum(r > market_avg_return for r in returns) / self._window
            if avg_return >= 0.95:
                top_stocks.append(symbol)

        top_stocks = top_stocks[: self._top_n]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest