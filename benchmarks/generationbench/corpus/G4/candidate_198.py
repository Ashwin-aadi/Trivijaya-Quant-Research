from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength6m(Strategy):
    rationale = (
        "This strategy selects stocks based on their 6-month relative strength against the Nifty 50 index. "
        "Historically, certain sectors like IT and healthcare show strong relative performance during economic downturns. "
        "By identifying top decile performers monthly and rebalancing quarterly, we aim to capitalize on sector-specific resilience."
    )

    def __init__(self, lookback: int = 180, top_decile_size: int = 20) -> None:
        self._lookback = lookback
        self._top_decile_size = top_decile_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        nifty_50_closes = history.select(
            pl.col("symbol").filter(pl.col("symbol") == "NIFTY 50").alias("nifty_close")
        )
        if nifty_50_closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        def relative_strength(row: pl.Series) -> float:
            stock_return = (row[-1] - row[0]) / row[0]
            nifty_return = (nifty_50_closes["nifty_close"].item() - row[0]) / row[0]
            return stock_return / nifty_return if nifty_return != 0 else 0

        ranked = history.groupby("symbol").agg(
            pl.col("adj_close").shift(1).list().alias("close_list")
        )
        ranked = ranked.with_columns((ranked["close_list"].apply(relative_strength)).alias("rs"))
        ranked = ranked.sort("rs", descending=True)
        top_stocks = [row[0] for row in ranked.to_pandas().head(self._top_decile_size).itertuples()]

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest