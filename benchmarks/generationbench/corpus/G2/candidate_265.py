from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion occurs when asset prices that have deviated from their historical "
        "mean tend to return to it over time. In a liquid market like NIFTY 100, mean "
        "reverting patterns can be observed due to the presence of noise traders and "
        "market participants who react to short-term volatility."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate the mean close for each symbol over the lookback period
        mean_closes: pl.DataFrame = (
            closes.groupby("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_close"))
                   .collect()
        )

        # Calculate the deviation of current close from the mean
        deviations: pl.DataFrame = (
            history.join(mean_closes, on="symbol", how="inner")
                    .with_columns((pl.col("adj_close") - pl.col("mean_close")).alias("deviation"))
                    .select(pl.exclude(["session_date", "symbol"]))
                    .collect()
        )

        # Identify symbols with large positive deviations
        top_deviators = deviations.top(k=self._window, order_by="deviation", descending=False)

        picks: list[str] = [row["symbol"] for _, row in top_deviators.iter_rows()]

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