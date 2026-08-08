from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for past winners to continue "
        "outperforming in the near term. By identifying symbols that have performed well "
        "relative to their peers recently, we can allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbol_counts = len(latest_closes.columns) - 1

        # Calculate returns
        returns = (
            history.select(pl.col("adj_close").shift(-self._window))
            .join(historical_closes := latest_closes.lazy(), on="symbol", how="inner")
            .with_columns(
                (pl.col("close") / pl.col(f"adj_close_{-1}") - 1.0).alias("recent_return"),
                (pl.col("close") / pl.col(f"adj_close_{-self._window}") - 1.0).alias("total_return"),
            )
        )

        # Filter and rank
        filtered_returns = returns.filter(pl.col("recent_return").is_not_null() & (pl.col("recent_return") > 0))
        if filtered_returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        ranked_returns = (
            filtered_returns.sort("recent_return", descending=True).head(self._window)
        )

        top_symbols = [row["symbol"] for row in ranked_returns.rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest