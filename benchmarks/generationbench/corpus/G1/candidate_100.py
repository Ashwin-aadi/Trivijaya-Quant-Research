from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAvg(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets whose prices have moved far "
        "away from their trailing average, we can generate entries when they return to more "
        "reasonable levels."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("avg_close"))
            .lazy()
            .collect()["avg_close"]
            .to_list()
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_close:
                continue
            history_sym = history.select(["session_date", "symbol", pl.col(symbol)]).filter(
                pl.col("symbol") == symbol
            )
            latest_close = float(view.latest_close()[symbol])
            trailing_avg = float(avg_close[symbol])

            price_diff = (latest_close - trailing_avg) / trailing_avg
            if abs(price_diff) > 0.1:  # Consider a threshold for reversion
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