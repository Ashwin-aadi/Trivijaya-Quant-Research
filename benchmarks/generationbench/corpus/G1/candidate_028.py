from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "long-term average are likely to return to it. This strategy aims to identify stocks "
        "that have fallen below a certain percentage of their 20-day moving average and "
        "allocate capital accordingly."
    )

    def __init__(self, window: int = 20, threshold: float = 0.95) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean"))
                   .with_columns((pl.col("adj_close") / pl.col("mean")).alias("deviation"))
        )

        latest_closes = view.closes(lookback=self._window)
        if latest_closes.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = (
            latest_closes.join(mean_prices, on="symbol", how="inner")
                         .select(pl.col("deviation"))
                         .to_dict(as_series=False)
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in symbol_prices or symbol_prices[symbol][-1] < self._threshold:
                picks.append(symbol)

        picks = picks[:5]
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