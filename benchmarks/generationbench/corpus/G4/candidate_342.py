from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy exploits the relative strength of stocks against a broad Indian equity index "
        "over a defined period. Stocks with higher returns compared to the index are expected to continue "
        "outperforming due to strong fundamentals or superior management."
    )

    def __init__(self, window: int = 90, top_n_percent: float = 20.0) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        index_close = view.history().select(pl.col("adj_close").alias("index_close"))[0]
        symbol_list = [symbol for symbol in view.symbols if symbol in closes.columns]

        returns_df = pl.DataFrame()
        for symbol in symbol_list:
            close_series = closes[symbol].drop_nulls().to_list()
            index_close_series = index_close.to_list()

            if len(close_series) < self._window or len(index_close_series) < self._window:
                continue

            log_prices = [pl.Series([float(p)]).log() for p in close_series[-self._window:]]
            log_index_prices = [pl.Series([float(p)]).log() for p in index_close_series[-self._window:]]

            relative_returns = [(p - log_prices[0]) / (log_index_prices[0] - log_prices[0]).exp().sum()
                                for p in log_prices]
            returns_df = pl.DataFrame({symbol: [float(r) for r in relative_returns]})

        if returns_df.height < 1:
            return Signal(information_available_at=stamp, weights={})

        ranked_stocks = (
            returns_df.mean()
            .sort("index", descending=True)
            .select(pl.col("index").rank(method="dense", descending=True))
        )

        top_n_count = int(len(symbol_list) * self._top_n_percent / 100.0)
        top_n_symbols = ranked_stocks.top_k(top_n_count).columns

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest