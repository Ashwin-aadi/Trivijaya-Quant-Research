from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price levels revert to a mean. By identifying symbols whose prices have deviated "
        "significantly from their trailing average and are now close to that level, we can "
        "anticipate a return towards the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.groupby("symbol")
            .agg((pl.col("adj_close").mean().alias("trailing_mean"),))
            .with_columns(
                (pl.col("close") / pl.col("trailing_mean") - 1).alias("deviation")
            )
        )

        # Identify symbols with significant deviation and close to the trailing mean
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in means.columns:
                continue
            latest_deviation = float(means.filter(pl.col("symbol") == symbol)[0, "deviation"])
            if -1.5 <= latest_deviation <= 1.5:
                picks.append(symbol)

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