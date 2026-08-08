from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion50d(Strategy):
    rationale = (
        "This strategy leverages price-level reversion against a trailing reference. It "
        "buys stocks when their prices deviate more than one standard deviation below their "
        "50-day simple moving average (SMA) and sells short when they cross above the SMA plus"
        " two standard deviations. Positions are exited based on either reaching within one "
        "standard deviation of the SMA or holding for 30 trading days."
    )

    def __init__(self, window: int = 50, threshold_long: float = -1.0, threshold_short: float = 2.0, exit_lookback: int = 30) -> None:
        self._window = window
        self._threshold_long = threshold_long
        self._threshold_short = threshold_short
        self._exit_lookback = exit_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select([pl.col("symbol"), pl.col("session_date").alias("date"), pl.col("adj_close")])
        sma = closes.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("sma")),
            (pl.col("adj_close").std().alias("std_dev"))
        )
        z_scores = (
            history
            .group_by("symbol")
            .with_columns(
                (pl.col("adj_close") - pl.col("sma")) / pl.col("std_dev").alias("z_score")
            )
        )

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in sma.columns or symbol not in z_scores.columns:
                continue

            latest_z_score = z_scores.select([pl.col(symbol).last().alias("latest_z_score")])[symbol].to_list()[0]
            latest_close = history.filter(pl.col("symbol") == symbol).select(pl.col("adj_close").last())[0, 0]

            if latest_z_score < self._threshold_long and (symbol not in signals):
                signals[symbol] = -0.02
            elif latest_z_score > self._threshold_short + sma.select([pl.col(symbol).alias("sma")])[symbol].to_list()[0]:
                signals[symbol] = 0.02

        if len(signals) == 0:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol, weight in signals.items()
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest