from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that securities with higher returns in the recent "
        "past tend to continue outperforming. This strategy invests in top performers based on "
        "historical returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        returns: pl.DataFrame = (
            history.drop_nulls()
                .group_by("symbol")
                .agg(
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                )
        )

        # Rank symbols by cumulative return over the window
        ranked: pl.DataFrame = (
            returns.groupby("symbol")
                   .agg((pl.col("return").sum()).alias("cumulative_return"))
                   .sort("cumulative_return", descending=True)
                   .head(self._top_n)
        )

        symbols_to_invest: list[str] = [r["symbol"] for r in ranked.to_dicts()]

        if not symbols_to_invest:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_invest)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_invest},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest