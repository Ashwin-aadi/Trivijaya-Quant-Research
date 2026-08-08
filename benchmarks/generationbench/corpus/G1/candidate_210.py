from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the broader market in recent days "
        "is a common approach to identify potentially strong performers. This strategy "
        "focuses on relative strength by comparing daily returns of each stock against "
        "the average return of the NIFTY 100 index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        avg_returns = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("avg_return"),
        ).group_by("symbol").agg(pl.col("avg_return").mean().alias("avg_return"))

        avg_nifty_returns = history.select(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("nifty_return")
        ).select(avg_nifty_return := pl.col("nifty_return").mean())

        nifty_avg_return = float(avg_nifty_returns["nifty_return"].to_list()[0])

        relative_strength = (
            history.select(
                pl.col("symbol").alias("symbol"),
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .join(avg_returns, on="symbol")
            .select(
                pl.col("symbol"),
                (pl.col("return") / pl.col("avg_return")) * nifty_avg_return.alias("relative_strength"),
            )
            .sort("relative_strength", descending=True)
        )

        top_symbols = [row["symbol"] for row in relative_strength.head(self._window).to_dict(as_pandas=False)["symbol"]]
        
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