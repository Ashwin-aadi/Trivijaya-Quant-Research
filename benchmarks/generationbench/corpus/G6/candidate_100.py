from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks with significant deviations from their historical "
        "average prices, focusing on mean reversion. It aims to capitalize on such opportunities "
        "while maintaining robust risk management practices suitable for the Indian equity market."
    )

    def __init__(self, window: int = 50, deviation_threshold: float = 2.0) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        sma_column = f"close_sma_{self._window}"
        std_dev_column = f"std_dev_{self._window}"

        # Calculate the 50-day simple moving average (SMA) and standard deviation
        history = (
            history.with_columns(
                pl.col("adj_close").rolling_mean(window=self._window).alias(sma_column),
                (pl.col("adj_close") - pl.col(f"{sma_column}")).pow(2).mean().alias(f"var_{self._window}"),
                pl.col(f"var_{self._window}").sqrt().alias(std_dev_column),
            )
            .drop([f"var_{self._window}", f"{sma_column}"])
            .sort("session_date", descending=False)
        )

        # Filter for stocks that close more than 2 standard deviations below their SMA by end-of-day close
        candidates = (
            history.with_columns(
                (pl.col("adj_close") - pl.col(sma_column)) / pl.col(std_dev_column).alias("deviation")
            )
            .filter(pl.col("deviation").lt(-self._deviation_threshold))
            .select(["symbol", "session_date", "adj_close", sma_column, std_dev_column])
        )

        if candidates.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Select top 20 stocks based on deviation
        top_symbols = [str(symbol) for symbol in candidates.sort("deviation", descending=True)["symbol"].head(20)]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest