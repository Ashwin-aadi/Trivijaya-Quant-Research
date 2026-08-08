from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on exploiting the liquidity premium by selecting and "
        "equally weighting stocks with high average daily trading volume (ADTV) and low bid-ask spreads. "
        "Highly liquid stocks tend to offer better price discovery and lower transaction costs, potentially leading to higher returns."
    )

    def __init__(self, window_adtv: int = 60, top_n: int = 20) -> None:
        self._window_adtv = window_adtv
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_adtv)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_measures = self._compute_liquidity_measures(history)

        # Select top N most liquid stocks
        top_n_symbols = [s for s in liquidity_measures.columns if s != "session_date"][: self._top_n]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_liquidity_measures(history: pl.DataFrame) -> pl.DataFrame:
    # Calculate Average Daily Trading Volume (ADTV)
    adtv = (
        history.group_by("symbol")
               .agg((pl.col("volume").sum() / self._window_adtv).alias("adtv"))
               .sort("adtv", descending=True)
    )

    # Calculate Bid-Ask Spread
    bid_ask_spread = (
        history.sort("session_date")
                .groupby_rolling(window=self._window_adtv, by="symbol", closed="both")
                .agg(
                    ((pl.col("high").max() - pl.col("low").min()) / 2).alias("bid_ask_spread"),
                )
    )

    # Combine and rank based on ADTV and bid-ask spread
    liquidity_measures = adtv.join(bid_ask_spread, on="symbol")
    liquidity_measures = (
        liquidity_measures.with_columns(
            (pl.col("adtv") + pl.col("bid_ask_spread")).rank(method="average", descending=True).alias("rank")
        )
               .sort("rank")
    )

    return liquidity_measures