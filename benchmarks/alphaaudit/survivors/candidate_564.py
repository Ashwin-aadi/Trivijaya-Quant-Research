from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks within an index to "
        "continue their recent performance. This strategy ranks stocks by recent returns and"
        " invests in the top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) == 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        returns_df = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date")
            .select(["symbol", "session_date", "r"])
        )

        # Calculate mean returns for each symbol
        mean_returns = (
            returns_df.group_by("symbol").agg(pl.col("r").mean().alias("mean_return"))
        )

        # Rank symbols by mean return
        ranked_symbols = (
            mean_returns.with_columns(
                (pl.col("mean_return").rank(method="ordinal", descending=True)).alias(
                    "rank"
                )
            )
            .sort("rank")
            .select(["symbol"])
        )

        top_symbols = [s.strip() for s in ranked_symbols["symbol"].to_list()[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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