from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy leverages the composite of two weakly related characteristics - earnings volatility and liquidity ratios - to identify undervalued stocks in the Indian market. By ranking stocks based on both metrics, we aim to capitalize on potential market sentiment or macroeconomic factors influencing these stocks simultaneously."
    )

    def __init__(self, vol_window: int = 60, vol_threshold: float = 15.0, liquidity_window: int = 30, liquidity_threshold: float = 0.02) -> None:
        self._vol_window = vol_window
        self._vol_threshold = vol_threshold
        self._liquidity_window = liquidity_window
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._vol_window + self._liquidity_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate earnings volatility
        vol_df = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .select(["symbol", "adj_close"])
            .with_column(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._vol_window) - 1.0).alias("returns")
            )
            .group_by("symbol")
            .agg(pl.col("returns").std().alias("volatility"))
        )

        # Calculate liquidity ratio
        liquidity_df = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .select(["symbol", "volume", "adj_close"])
            .with_column((pl.col("volume") / pl.col("adj_close")).alias("liquidity_ratio"))
            .group_by("symbol")
            .agg([pl.mean("liquidity_ratio").alias("avg_liquidity_ratio")])
        )

        # Merge and rank companies based on both signals
        combined_df = vol_df.join(liquidity_df, on="symbol")
        ranked_companies = (
            combined_df.sort(
                (pl.col("volatility") < self._vol_threshold) & (pl.col("avg_liquidity_ratio") > self._liquidity_threshold),
                descending=True,
            )
            .head(30)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not ranked_companies:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_companies)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_companies},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest