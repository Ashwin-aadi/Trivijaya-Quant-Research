from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that have "
        "experienced extreme price movements relative to their historical volatility. It uses "
        "Bollinger Bands and Average True Range (ATR) to define entry points, ensuring timely "
        "execution before the price returns to its mean."
    )

    def __init__(self, window_sma: int = 20, window_atr: int = 14, top_n: int = 30) -> None:
        self._window_sma = window_sma
        self._window_atr = window_atr
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_sma, self._window_atr))
        if history.is_empty() or history.height < max(self._window_sma, self._window_atr):
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        def rank_bollinger_band(row: pl.Series) -> float:
            close = row[1]
            sma = row[-2]
            lower_band = row[-4]  # Assuming the lower band is at -2 standard deviations
            upper_band = row[-3]  # Assuming the upper band is at +2 standard deviations
            if lower_band < close < sma:
                return (sma - close) / (upper_band - lower_band)
            elif sma < close < upper_band:
                return -(close - sma) / (upper_band - lower_band)
            else:
                return 0.0

        bollinger_bands = (
            history.sort("session_date")
            .group_by("symbol")
            .agg(
                [
                    pl.col("adj_close").mean().alias("sma"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).std()
                    .over(pl.arange(1, self._window_sma + 1)).alias("std_dev"),
                    (pl.col("sma").shift(-self._window_sma) - pl.col("sma")).alias("sma_shift"),
                ]
            )
        )

        filtered_bands = bollinger_bands.filter(
            (bollinger_bands["return"] > 0.15) | (bollinger_bands["return"] < -0.15)
        ).select(["symbol", "sma", "std_dev", "sma_shift"])

        if filtered_bands.height == 0:
            return Signal(information_available_at=stamp, weights={})

        atr = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).max(pl.col("close") - pl.col("low"))
                .max(pl.col("high") - pl.col("close")).alias("true_range")
            )
            .group_by("symbol")
            .agg([(pl.col("true_range").mean()).alias("atr_mean")])
        )

        final_df = (
            filtered_bands.join(atr, on="symbol", how="inner")
            .sort("symbol")
            .select(["symbol", "sma", "std_dev", "sma_shift", "atr_mean"])
            .with_columns(
                (pl.col("sma") - pl.col("sma_shift")) / pl.col("std_dev").alias("rank_band"),
                ((2 * pl.col("atr_mean")).is_null()).alias("low_volatility")
            )
        )

        final_df = (
            final_df.with_column((pl.when(final_df["low_volatility"])
                                  .then(0.0)
                                  .otherwise(final_df["rank_band"])).alias("final_rank"))
            .sort("symbol", descending=True)
            .tail(self._top_n)
            .select(["symbol"])
        )

        if final_df.height == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(final_df)
        weights = {s: weight for s in final_df["symbol"].to_list()}
        return Signal(
            information_available_at=stamp,
            weights={k: float(v) for k, v in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest