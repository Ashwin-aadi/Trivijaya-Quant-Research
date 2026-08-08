from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "historical average should revert. This strategy identifies stocks where the price has "
        "dropped below a trailing moving average and allocates capital accordingly."
    )

    def __init__(self, window: int = 50, threshold: float = 1.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.group_by("symbol").agg(
            (pl.col("adj_close").mean().alias("trailing_mean"))
        )
        latest_closes = view.closes()
        combined = (
            latest_closes.join(mean_close, on="symbol", how="left")
                          .with_columns((pl.col("close") / pl.col("trailing_mean") - 1.0).alias("reversion_ratio"))
                          .sort("reversion_ratio", descending=True)
                          .head(5)
        )

        symbols = combined["symbol"].to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest