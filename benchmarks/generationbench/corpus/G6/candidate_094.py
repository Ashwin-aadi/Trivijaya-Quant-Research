from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalitySignalStrategy(Strategy):
    rationale = (
        "Stock markets often exhibit seasonality driven by fiscal years, agricultural cycles, "
        "and government policies. By identifying months with historically higher returns and using a"
        " moving average crossover strategy, we can capture these seasonal effects to generate trading signals."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_20 = history.select(
            pl.col("adj_close").rolling_mean(self._window).alias(f"sma_{self._window}")
        )
        macd = (sma_9 := sma_20.select(pl.col(f"sma_{self._window}").shift(8)).fill_null(
            0
        )).select(
            (pl.col(f"sma_{self._window}") - sma_9[f"sma_{self._window}"]).alias("macd")
        )
        macd_signal = macd.select(
            pl.col("macd").rolling_mean(9, weights=[1 / 9] * 9).alias("signal")
        )
        history = history.join(macd_signal, on="session_date")

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            macd_values = [float(v) for v in history[symbol + "_macd"].drop_nulls().to_list()]
            signal_values = [
                float(v) for v in history[symbol + "_signal"].drop_nulls().to_list()
            ]
            if len(macd_values) < self._window or len(signal_values) < 9:
                continue
            crossover_points = [
                i
                for i, (macd_val, signal_val) in enumerate(zip(macd_values, signal_values))
                if macd_val > signal_val and macd_values[i - 1] <= signal_values[i - 1]
            ]
            if crossover_points:
                picks.append(symbol)

        picks = picks[: self._top_n]
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