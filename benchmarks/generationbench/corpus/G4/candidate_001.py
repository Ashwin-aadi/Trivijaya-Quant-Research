from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BollingerReversion(Strategy):
    rationale = (
        "Short-term price reversions in stock prices within the Indian market are exploited "
        "by identifying undervalued and overvalued stocks based on their 20-day Bollinger Bands. "
        "The strategy aims to capitalize on temporary mispricings before prices revert to historical norms."
    )

    def __init__(self, window: int = 20, std_devs: float = 2) -> None:
        self._window = window
        self._std_devs = std_devs

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma = history.select(
            pl.col("adj_close").mean().alias("sma")
        ).with_columns(
            (pl.col("adj_close") - pl.col("sma")).alias("deviation"),
            ((pl.col("adj_close") - pl.col("sma")) / self._std_devs).alias("z_score")
        )

        def bollinger_bands(row: pl.Series) -> tuple[float, float]:
            mean = row["sma"]
            std_dev = row.std()
            upper_band = mean + (self._std_devs * std_dev)
            lower_band = mean - (self._std_devs * std_dev)
            return (lower_band, upper_band)

        bollinger_bands_df = sma.groupby("session_date").agg(
            bollinger_bands(pl.col("adj_close")).alias(("lower_band", "upper_band"))
        )

        closes = view.closes(lookback=self._window)
        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            z_score = sma.select(pl.col(symbol).tail(self._window)["z_score"]).to_series().mean()
            lower_band = bollinger_bands_df.select(pl.col("lower_band").max()).to_series()[0]
            upper_band = bollinger_bands_df.select(pl.col("upper_band").min()).to_series()[0]

            if z_score < -1.5 and closes.select(symbol)[-1] <= lower_band:
                signals[symbol] = 1.0 / len(signals)
            elif z_score > 1.5 and closes.select(symbol)[-1] >= upper_band:
                signals[symbol] = -1.0 / len(signals)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series()[0]
    assert isinstance(newest, date)
    return newest