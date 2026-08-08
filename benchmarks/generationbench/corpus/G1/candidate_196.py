from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reverts towards the mean over time. By tracking a trailing average and "
        "identifying deviations from it, we can find potential entry points when prices "
        "move back to normal levels."
    )

    def __init__(self, window: int = 50, deviation_threshold: float = 1.5) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbols = list(latest_closes.keys())

        # Calculate trailing mean and standard deviation
        prices = history.select(
            pl.col("symbol").alias("symbol"),
            pl.col("adj_close").rolling_mean(self._window).alias("trailing_mean"),
            (pl.col("adj_close") - pl.col("adj_close").rolling_mean(self._window)).pow(2)
            .rolling_sum(self._window)
            / self._window
            .sqrt()
            .alias("std_dev")
        )

        # Filter out symbols without sufficient data
        prices = prices.filter(
            (pl.col("symbol").is_in(symbols))
            & (
                pl.col("trailing_mean").is_not_null()
                & pl.col("std_dev").is_not_null()
            )
        )

        reversion_candidates = []
        for symbol in symbols:
            price_history = history.select(
                "session_date",
                (pl.col("adj_close") - latest_closes[symbol])
                / latest_closes[symbol]
                .alias(f"return_{symbol}"),
                pl.col("trailing_mean").filter(pl.col("symbol") == symbol).alias("mean"),
                pl.col("std_dev").filter(pl.col("symbol") == symbol).alias("std")
            )
            # Calculate z-score
            price_history = price_history.with_columns(
                (pl.col(f"return_{symbol}") - pl.col("mean"))
                / pl.col("std")
                .alias("z_score")
            )

            if (
                price_history.select(pl.col("z_score").max()).item()
                > self._deviation_threshold
            ):
                reversion_candidates.append(symbol)

        weights = {s: 1.0 / len(reversion_candidates) for s in reversion_candidates}
        return Signal(
            information_available_at=stamp, weights={s: weights[s] for s in symbols if s in weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest