from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion occurs when prices return to a previous level after deviating from it. "
        "A trailing reference price could be the 20-day moving average of closing prices. "
        "When current prices deviate significantly from this trailing average, they are likely "
        "to revert back towards it, providing an opportunity for profitable trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._window)

        # Calculate the 20-day moving average of closing prices for each symbol
        ma = (
            history.select(pl.col("adj_close").rolling_mean(self._window))
            .with_columns(
                pl.col("session_date").alias("date"),
                [pl.lit(s) for s in symbols].alias("symbol"),
            )
            .unnest("adj_close")
            .pivot(index="date", columns="symbol", values="mean_adj_close")
        )

        # Calculate the deviation of current close from the trailing moving average
        dev = closes.join(ma, on="session_date", how="left").with_columns(
            (pl.col("adj_close") - pl.col(f"close_{self._window}")).alias("deviation")
        )

        if dev.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Identify symbols with high deviation
        top_devs = dev.sort("deviation", descending=True).select(
            [pl.col(s) for s in symbols]
        ).rows()[-5:]

        picks: list[str] = [symbol for symbol, _ in top_devs]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest