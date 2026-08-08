from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeFeatureStrategy(Strategy):
    rationale = (
        "This strategy combines two simple indicators: the 20-day moving average and the "
        "5-day RSI. The idea is that a stock breaking above its 20-day moving average and having "
        "a positive 5-day RSI might indicate strong upward momentum."
    )

    def __init__(self, ma_window: int = 20, rsi_window: int = 5) -> None:
        self._ma_window = ma_window
        self._rsi_window = rsi_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._ma_window + self._rsi_window)
        if history.height < self._ma_window + self._rsi_window:
            return Signal(information_available_at=stamp, weights={})

        ma_column_name = f"ma_{self._ma_window}"
        rsi_column_name = f"rsi_{self._rsi_window}"

        # Compute 20-day moving average
        history = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._ma_window)).alias(ma_column_name)
            )
            .with_columns((pl.col("close") / pl.col(ma_column_name) - 1.0).alias("ma_deviation"))
        )

        # Compute 5-day RSI
        history = (
            history.with_columns(
                (pl.col("adj_close").diff().sign() * 100)
                .rolling_sum(window_size=self._rsi_window, by="symbol")
                .alias(rsi_column_name)
            )
            .with_columns((pl.col(rsi_column_name) / self._rsi_window).alias("normalized_rsi"))
        )

        # Select symbols that break above the 20-day MA and have a positive RSI
        picks: list[str] = []
        for symbol in view.symbols:
            if (
                float(history.filter(pl.col("symbol") == symbol)["ma_deviation"].to_list()[-1]) > 0.0
                and float(history.filter(pl.col("symbol") == symbol)[rsi_column_name].to_list()[-1]) > 0.0
            ):
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest