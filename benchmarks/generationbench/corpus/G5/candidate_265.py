from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and returns eventually return to the long-term "
        "mean. By identifying stocks that have significantly deviated from their mean price over a "
        "short period, we can capitalize on their likely return to equilibrium."
    )

    def __init__(self, window: int = 20, threshold: float = 0.15) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean_price")
        )
        closes = view.closes()

        # Calculate deviations from the mean
        deviations = (
            closes.join(mean_close, on="symbol", how="left")
            .with_columns(
                (
                    pl.col("adj_close") - pl.col("mean_price")
                ).alias("deviation"),
                (pl.col("adj_close") / pl.col("mean_price") - 1.0).alias("percent_deviation")
            )
            .filter(pl.col("deviation").is_not_null())
        )

        if deviations.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Select symbols with significant deviation from mean
        picks: list[str] = []
        for symbol in view.symbols:
            value = float(deviations.filter(pl.col("symbol") == symbol)["deviation"].item())
            if abs(value) > self._threshold:
                picks.append(symbol)

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