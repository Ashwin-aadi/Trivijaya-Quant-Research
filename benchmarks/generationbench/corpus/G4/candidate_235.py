from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeDispersionStrategy(Strategy):
    rationale = (
        "This strategy identifies periods of market range compression and dispersion to "
        "capitalize on stock price movements. During compression, it selects stocks with "
        "technical indicators suggesting potential breakouts. Conversely, during dispersion, "
        "it reduces exposure to minimize risk from heightened volatility."
    )

    def __init__(self, window_compression: int = 60, threshold_compression: float = 0.15,
                 window_dispersion: int = 30, threshold_dispersion: float = 0.25) -> None:
        self._window_compression = window_compression
        self._threshold_compression = threshold_dispersion
        self._window_dispersion = window_dispersion
        self._threshold_dispersion = threshold_dispersion

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_compression + self._window_dispersion)

        if closes.height < self._window_compression + self._window_dispersion:
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day standard deviation of daily returns using open price
        hist = view.history(lookback=self._window_compression + self._window_dispersion)
        opens = hist.select(pl.col("symbol").alias("symbol"), pl.col("open").alias("open"))
        closes = closes.join(opens, on="symbol", how="left")
        returns = (closes["adj_close"] / closes["open"].shift(1) - 1.0).alias("return")
        hist = closes.with_columns(returns)
        vol_20d = hist.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("return").rank(method="dense", descending=False)).alias("rank_return")
        ).group_by("symbol").agg(pl.col("rank_return").std().alias("volatility"))

        # Identify range compression and dispersion
        compress_signal = vol_20d.filter(
            pl.col("volatility") < self._threshold_compression
        )
        disp_signal = vol_20d.filter(
            pl.col("volatility") > self._threshold_dispersion
        )

        if compress_signal.is_empty() and disp_signal.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Technical indicators: 50-day and 200-day SMA of closing prices
        sma_50d = hist.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(49) - 1.0).alias("sma_50d")
        ).group_by("symbol").agg((pl.col("sma_50d").mean().alias("mean_sma_50d"),
                                  (pl.col("sma_50d").sum().rank(method="dense", descending=False)).alias("rank_sma_50d")))

        # Relative Strength Index (RSI) for 14 days
        rsi_14d = hist.select(
            pl.col("symbol").alias("symbol"),
            ((pl.col("adj_close") - pl.col("adj_close").shift(13)) / pl.col("adj_close").shift(13)).alias("rsi_14d")
        ).group_by("symbol").agg((pl.col("rsi_14d").mean().alias("mean_rsi_14d"),
                                  (pl.col("rsi_14d").rank(method="dense", descending=False)).alias("rank_rsi_14d")))

        # Combine signals and rank stocks
        combined = compress_signal.join(sma_50d, on="symbol", how="left").join(rsi_14d, on="symbol", how="left")
        if not disp_signal.is_empty():
            disp_combined = disp_signal.join(sma_50d, on="symbol", how="left").join(rsi_14d, on="symbol", how="left")
        else:
            disp_combined = pl.DataFrame()

        # Range compression: select top 20 stocks
        if not combined.is_empty():
            combined = combined.with_columns(
                (pl.col("mean_sma_50d") - pl.col("rank_sma_50d")).alias("rank_gap"),
                (pl.col("rank_rsi_14d")).alias("rank_rsi")
            ).sort("rank_gap", descending=True).sort("rank_rsi", descending=True)
            top_stocks = combined.head(20)["symbol"].to_list()
            weight = 1.0 / len(top_stocks) if top_stocks else 0.0
        else:
            top_stocks, weight = [], 0.0

        # Dispersion: exit or hedge positions
        if not disp_combined.is_empty():
            disp_combined = disp_combined.with_columns(
                (pl.col("mean_sma_50d") - pl.col("rank_sma_50d")).alias("rank_gap"),
                (pl.col("rank_rsi_14d")).alias("rank_rsi")
            ).sort("rank_gap", descending=True).sort("rank_rsi", descending=True)
            exit_stocks = disp_combined.head(20)["symbol"].to_list()
        else:
            exit_stocks = []

        # Generate weights
        if top_stocks and not exit_stocks:
            return Signal(
                information_available_at=stamp, weights={s: weight for s in top_stocks}
            )
        elif not top_stocks and exit_stocks:
            # Exit positions during dispersion phase
            return Signal(
                information_available_at=stamp, weights={s: -0.15 / len(exit_stocks) for s in exit_stocks}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest