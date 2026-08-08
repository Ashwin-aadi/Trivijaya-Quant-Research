from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the tendency for stocks that have outperformed "
        "in the past to continue outperforming in the future. This is based on the idea that "
        "strong performance in a stock can indicate it is undervalued, and therefore worth investing in."
    )

    def __init__(self, lookback_period: int = 60, top_n: int = 10) -> None:
        self._lookback_period = lookback_period
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.is_empty() or len(history.columns) < 2 + view.symbols[0]:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).sort("session_date")

        # Group by symbol and calculate cumulative return over the lookback period
        cum_returns = (
            history.groupby("symbol", maintain_order=True)
            .agg(
                (pl.col("returns").sum()).alias("cumulative_return"),
                pl.col("adj_close").last().alias("latest_close"),
            )
            .sort("cumulative_return", descending=True)
            .head(self._top_n + 1)  # Include all symbols for now
        )

        top_symbols = [row["symbol"] for row in cum_returns.to_dicts()]
        weights = {s: 1.0 / len(top_symbols) if s in top_symbols else 0.0 for s in view.symbols}

        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if v > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest