from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow long-term trends while scaling trade sizes based on recent volatility. "
        "Higher positions are taken during periods of low volatility, and risk is reduced during high volatility."
    )

    def __init__(self, window: int = 50, atr_window: int = 14, max_positions: int = 30) -> None:
        self._window = window
        self._atr_window = atr_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window).with_columns(
            (pl.col("symbol") == pl.col(history["symbol"][-1])).alias("is_latest")
        )
        if closes.height < self._max_positions + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate Simple Moving Average
        sma = history.select(
            pl.col("adj_close").rolling_mean(self._window).alias(f"sma_{self._window}")
        )

        # Calculate True Range and ATR
        tr = (pl.col("high") - pl.col("low")).alias("tr")
        atr = (
            sma.join(tr, on="session_date", how="inner")
                .select(
                    (pl.col("tr").rolling_mean(self._atr_window).alias(f"atr_{self._atr_window}"))
                )
        )

        # Combine SMAs and ATR
        combined = history.join(atr, on="session_date", how="inner")
        combined = combined.with_columns(
            (
                pl.col("adj_close") - pl.col(f"sma_{self._window}")
            ).abs().alias("distance_from_sma"),
            (pl.col("atr_" + str(self._atr_window))).alias("atr")
        )

        # Filter and rank symbols based on trend strength and volatility
        ranked = combined.select(
            [
                "symbol",
                ("distance_from_sma" / self._window).alias("trend_strength"),
                "atr"
            ]
        ).sort(
            ["trend_strength", "atr"], descending=[True, True]
        ).head(self._max_positions)

        weights: dict[str, float] = {}
        if not ranked.is_empty():
            for _, row in ranked.iter_rows():
                symbol = row["symbol"]
                atr_value = row["atr"]
                weight = 0.01 / (2 * (atr_value / self._atr_window) + 1)
                weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest