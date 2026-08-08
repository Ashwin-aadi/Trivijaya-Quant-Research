from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy exploits the phenomenon of dispersion or range compression in stock prices. "
        "It identifies stocks experiencing high volatility and waits for a period of low volatility to enter positions."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate Bollinger Bands
        bollinger_bands = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._window)).alias("sma"),
                ((pl.col("adj_close") - pl.col("sma")) / 2 * pl.col("volume")).sum().alias("volatility")
            )
            .select(
                ["symbol", "session_date", "close", "sma", (pl.col("close") - 2 * pl.col("volatility")).alias("lower_band"),
                 (pl.col("close") + 2 * pl.col("volatility")).alias("upper_band")]
            )
            .collect()
        )

        # Calculate RSI
        rsi = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close").diff().abs()).rolling_sum(window_size=self._window - 1).alias("gain"),
                ((pl.col("adj_close").shift(-1) - pl.col("adj_close")).clip(0, None)).rolling_sum(window_size=self._window - 1).alias("loss")
            )
            .with_columns(
                (100 * (pl.col("gain") / (pl.col("gain") + pl.col("loss")))).alias("rsi")
            )
            .select(["symbol", "session_date", "close", "rsi"])
            .collect()
        )

        # Filter stocks based on Bollinger Band width and RSI
        filtered = (
            bollinger_bands.join(rsi, on=["symbol", "session_date"], how="inner")
            .with_columns(
                ((pl.col("upper_band") - pl.col("lower_band")) / 2).alias("band_width"),
                (pl.col("rsi").between(30, 70)).cast(pl.int8).alias("in_range")
            )
            .filter((pl.col("band_width").rank(method="dense", descending=True) <= self._top_n)
                    & (pl.col("in_range") == 1))
        )

        if filtered.height == 0:
            return Signal(information_available_at=stamp, weights={})

        picks = filtered["symbol"].to_list()[:self._top_n]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest