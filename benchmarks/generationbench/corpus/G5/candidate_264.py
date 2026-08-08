from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed in "
        "the recent past to continue to outperform. This strategy ranks stocks by their returns "
        "over a short period and allocates capital accordingly, while considering liquidity constraints."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .select(["symbol", "session_date", "return"])
        )

        # Filter out null returns
        history = history.filter(pl.col("return").is_not_null())

        # Rank by return, considering liquidity constraints
        ranks = (
            history.group_by("symbol")
            .agg(
                (pl.col("return").rank(method="ordinal", descending=True).alias("rank")),
                (pl.col("volume").mean().alias("avg_volume"))
            )
            .sort("rank")
        )

        # Filter out symbols with very low liquidity
        min_avg_volume = 10_000  # Adjust based on market conditions
        ranks = ranks.filter(pl.col("avg_volume") > min_avg_volume)

        top_symbols = [row.symbol for row in ranks.rows()][:self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest