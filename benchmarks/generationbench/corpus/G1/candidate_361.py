from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that a security's price will eventually reverse after "
        "deviating significantly from its mean. By identifying symbols that have deviated"
        " most from their recent average, we can generate trades expecting a return to the mean."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .select(["symbol", "mean"])
        )
        recent_closes = view.closes(lookback=self._window)
        symbol_means: list[str] = []
        for symbol in view.symbols:
            if symbol not in mean_close["symbol"].to_list():
                continue
            mean_val = float(mean_close.filter(pl.col("symbol") == symbol)["mean"][0])
            recent_close_values = [float(v) for v in recent_closes[symbol].drop_nulls().to_list()]
            deviation = abs(recent_close_values[-1] - mean_val)
            if deviation >= max([abs(val - mean_val) for val in recent_close_values]):
                symbol_means.append(symbol)

        if not symbol_means:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_means)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbol_means},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest