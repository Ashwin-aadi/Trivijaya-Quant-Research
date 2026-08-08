from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit higher returns during specific times of the year. "
        "By identifying and capitalizing on these seasonal effects, we can construct a strategy "
        "that leverages historical patterns for better performance."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Group by symbol and get the maximum close for each month
        monthly_max_closes = (
            closes.group_by(pl.date.Month(view.as_of))
            .agg(pl.col("adj_close").max().alias("monthly_max"))
            .select(
                ["symbol", "session_date", pl.col("monthly_max").alias("max_close")]
            )
        )

        # Filter the symbols based on their maximum close
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in monthly_max_closes.columns:
                continue
            max_close = float(monthly_max_closes[monthly_max_closes["symbol"] == symbol]["max_close"])
            latest_close = float(view.latest_close()[symbol])
            if latest_close >= max_close:
                picks.append(symbol)

        picks = picks[: self._top_n]
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