from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Prices that revert to a trailing mean suggest that recent price deviations are "
        "part of normal market noise. This strategy aims to capture the reversionary effect, "
        "buying stocks that have deviated below their mean and selling those that have risen above it."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("trailing_mean"))
            .with_column(pl.col("trailing_mean").shift(1).alias("previous_mean"))
            .select(["symbol", "previous_mean"])
            .collect()
        )

        latest_closes = view.closes().select(symbols)
        price_diffs = (
            latest_closes.join(mean_close, on="symbol")
            .with_column(
                (pl.col("adj_close") - pl.col("previous_mean"))
                .alias("price_diff")
            )
        )

        signals: dict[str, float] = {}
        for symbol in symbols:
            price_diff = price_diffs.get_column(symbol).to_list()[0]
            if price_diff > 2 * history.select([symbol]).height / self._window:
                signals[symbol] = -1.0
            elif price_diff < -2 * history.select([symbol]).height / self._window:
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items() if w != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest