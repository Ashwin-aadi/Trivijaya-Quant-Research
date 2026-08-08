from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityLiquidityStrategy(Strategy):
    rationale = (
        "This strategy exploits the composite characteristics of daily price volatility and "
        "average daily trading volume (ADTV) to identify stocks with high volatility but low liquidity. "
        "Such imbalances can lead to frequent price movements that are potentially exploitable."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily price volatility
        history = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("price_diff"),
                ((pl.col("close").shift(-1) - pl.col("adj_close")) / pl.col("adj_close")).alias("return"),
                (pl.col("price_diff").std()).over(pl.col("session_date")).alias("volatility")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(
                pl.col("close").mean().alias("recent_adj_close"),
                (pl.col("return").sum() / self._window).alias("total_return"),
                (pl.col("price_diff").std()).alias("volatility"),
                (pl.col("volume").mean()).alias("avg_volume")
            )
        )

        # Calculate liquidity metric
        liquidity = view.history(lookback=self._window)
        liquidity = (
            liquidity.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume")
            )
        ).select(["symbol", "avg_volume"])

        # Combine volatility and liquidity scores
        history = history.join(liquidity, on="symbol", how="inner")
        combined_score = 0.7 * (history["volatility"] / history["volatility"].max()) + 0.3 * (1 - history["avg_volume"] / history["avg_volume"].max())
        history = history.with_columns(combined_score.alias("combined_score"))

        # Rank the stocks based on combined score
        ranked_stocks = history.sort("combined_score", descending=True).select(["symbol", "combined_score"])

        if len(ranked_stocks) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = [row["symbol"] for _, row in ranked_stocks.to_dicts()[:self._top_n]]
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest