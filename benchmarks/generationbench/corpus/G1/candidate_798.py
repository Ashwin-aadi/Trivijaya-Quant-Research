from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed the broader market in recent periods are more likely to continue "
        "outperforming due to momentum effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        market_close = view.closes().select(
            pl.col(view.as_of).alias("market_close")
        )
        market_returns = (view.closes() / market_close.shift(1) - 1.0).with_columns(
            pl.when(pl.all().is_nan()).then(0.0)
        )

        symbol_returns = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )
        avg_market_return = market_returns.mean()
        avg_symbol_returns = symbol_returns.groupby("symbol").agg(
            pl.col("r").mean().alias("avg_r")
        )

        outperforming_symbols = (
            avg_symbol_returns.with_columns(
                (pl.col("avg_r") / avg_market_return * 100.0).alias("outperformance")
            )
            .sort("outperformance", descending=True)
            .select(pl.col("symbol"))
            .head(5)
        )

        if outperforming_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(outperforming_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in outperforming_symbols["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest