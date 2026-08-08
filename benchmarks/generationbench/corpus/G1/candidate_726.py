from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks relative to the NIFTY 100 index "
        "based on their cumulative returns over a lookback period. The idea is that "
        "stocks with higher relative strength are more likely to continue outperforming."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        index_returns = (
            view.closes(lookback=self._window)
            .select(pl.col("adj_close").last())
            / pl.col("adj_close")
            - 1.0
        )
        index_returns = (index_returns * 100).round(2)

        stock_returns = history.with_columns(
            ((pl.col("close") / pl.col("close").shift(1) - 1.0) * 100).alias("return")
        ).sort("session_date", descending=True)
        stock_returns = stock_returns.select(pl.col("symbol"), "return")

        combined = (
            stock_returns.join(index_returns, on="session_date", how="left")
            .group_by("symbol")
            .agg(
                (pl.col("return").mean().alias("avg_return")),
                pl.col("adj_close").last().alias("latest_price"),
                pl.col("index_return").first().alias("index_return"),
            )
        )

        rank = combined.with_columns(
            (
                (pl.col("avg_return") - pl.col("index_return")) / 2
            ).rank(method="ordinal", descending=True).alias("strength_rank")
        )

        top_symbols = [row["symbol"] for row in rank.sort("strength_rank").select("symbol").rows()[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in top_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest