from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5to10(Strategy):
    rationale = (
        "This strategy exploits mean reversion in short-term equity prices by identifying "
        "stocks that are currently underpriced relative to their recent historical averages. "
        "By buying such stocks and selling overpriced ones, the strategy aims to profit from "
        "price corrections towards historical norms."
    )

    def __init__(self, window: int = 10, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sma5to10 = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().over("session_date").shift(-self._window).alias(f"m_{self._window}"))
            )
        )

        closes = view.closes(lookback=self._window)
        deviations = (
            sma5to10
            .join(closes, on="symbol", how="left")
            .with_columns(
                (pl.col("adj_close") - pl.col(f"m_{self._window}")).abs().alias("deviation"),
            )
            .sort("deviation", descending=True)
        )

        if deviations.is_empty():
            return Signal(information_available_at=stamp, weights={})

        ranks = (
            deviations
            .select([pl.col("symbol"), pl.col("deviation").rank(descending=False)])
            .with_columns(pl.when(pl.col("deviation") < 0).then(-1).otherwise(1).alias("sign"))
            .group_by("symbol")
            .agg((pl.col("sign") * (pl.col("deviation") / abs(pl.col("deviation")).max())).sum().alias("rank"))
        )

        top_20_percent = int(len(view.symbols) * 0.2)
        buys = ranks.filter(pl.col("rank") <= -top_20_percent).select("symbol").to_list()
        sells = ranks.filter(pl.col("rank") >= top_20_percent).select("symbol").to_list()

        buy_weights, sell_weights = {}, {}
        for symbol in buys:
            buy_weights[symbol] = 1.0 / len(buys) * 0.6
        for symbol in sells:
            sell_weights[symbol] = -1.0 / len(sells) * 0.4

        return Signal(
            information_available_at=stamp,
            weights={**buy_weights, **sell_weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest