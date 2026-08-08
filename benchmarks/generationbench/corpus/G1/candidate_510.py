from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks relative to the NIFTY 100 index "
        "over a certain period. The idea is that stocks outperforming the market may offer "
        "greater returns and are less susceptible to broad market declines."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < (self._window * len(view.symbols)):
            return Signal(information_available_at=stamp, weights={})

        index_returns = (
            view.closes()
            .select("session_date")
            .join(history.select(pl.col("symbol"), pl.col("adj_close").last()), on="symbol", how="left")
            .with_columns(
                (pl.col("close_x") / pl.col("close_y") - 1).alias("index_return"),
            )
        )

        stock_returns = (
            history
            .select(pl.col("symbol"), "session_date", "adj_close")
            .pivot(index="symbol", values="adj_close", aggregate_function=None)
            .with_columns(
                (pl.col(symbol) / pl.col(symbol).shift(1) - 1).alias(f"return_{symbol}")
                for symbol in view.symbols
            )
        )

        merged = index_returns.join(stock_returns, on="session_date")
        
        # Calculate relative strength as the ratio of stock return to index return
        merged = (
            merged
            .with_columns(
                (pl.col(f"return_{symbol}") / pl.col("index_return")).alias(f"relative_strength_{symbol}")
                for symbol in view.symbols
            )
            .sort(pl.col([f"relative_strength_{symbol}" for symbol in view.symbols]), descending=True)
            .select([pl.col("session_date"), *[f"relative_strength_{symbol}" for symbol in view.symbols]])
        )

        top_n = min(self._window, len(view.symbols))
        strongest_symbols = merged.columns[1:top_n+1]

        weight = 1.0 / len(strongest_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol.split("_")[2]: weight for symbol in strongest_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest