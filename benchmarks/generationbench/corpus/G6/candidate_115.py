from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class AtrDispersionStrategy(Strategy):
    rationale = (
        "This strategy capitalizes on periods of increased volatility (dispersion) or reduced price action (range compression) in the Indian equity market by leveraging the Average True Range (ATR). "
        "High ATR indicates heightened volatility, while low ATR suggests range compression. "
        "Entries are made based on these conditions with corresponding exit rules to manage risk."
    )

    def __init__(self, window: int = 20, std_dev_threshold: float = 1.0, consecutive_days: int = 3) -> None:
        self._window = window
        self._std_dev_threshold = std_dev_threshold
        self._consecutive_days = consecutive_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        atr = (history.high - history.low).mean().over("symbol").alias("atr")

        # Calculate ATR for each symbol over the lookback period
        df = (
            history.with_columns(
                pl.col("high").diff().abs().max(pl.col("low").diff().abs()).max(pl.col("close").shift(-1) - pl.col("open")).alias("tr"),
                (pl.col("tr").rolling_mean(window_size=self._window)).alias("atr")
            )
        )

        # Filter symbols for high and low ATR
        df = (
            df.filter(
                ((df.atr > df.atr.mean() + df.atr.stddev()) & (df.session_date == stamp - date(0, 0, self._window))) |
                (pl.col("session_date").rolling_max().over("symbol", window_size=self._consecutive_days) < pl.col("atr").shift(-self._consecutive_days))
            )
        )

        # Select top symbols for entries
        picks: list[str] = df.select(["symbol"]).to_series().to_list()[:7]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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