from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for assets that have outperformed in "
        "the recent past to continue to outperform. This is based on the idea that stocks with "
        "strong positive returns are more likely to maintain their momentum, while those with "
        "negative returns are less likely to continue underperforming."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        mean_returns = (
            (closes.select(pl.col("adj_close").reverse().head(self._window))
             .select((pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("return"))
             .select(pl.col("return").mean())
             .to_series()
             .to_list()[0])
        )

        momentum_ranks = history.select(
            (pl.col("symbol"), (pl.col("close") / mean_returns).rank(method="ordinal", descending=True))
        ).collect().with_column(pl.col("symbol").alias("symbol_rank"))

        top_symbols = (
            momentum_ranks.sort("symbol_rank")
            .select(["symbol"])
            .head(self._top_n)
            .to_dict(as_series=False)["symbol"]
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest