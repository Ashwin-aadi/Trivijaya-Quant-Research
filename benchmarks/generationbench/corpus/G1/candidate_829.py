from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies look for assets that have deviated significantly from their "
        "historical mean prices and expect the price to revert back. Short-horizon mean reversion "
        "can be particularly effective in volatile markets."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            adj_closes = history.select(pl.col("adj_close")).filter(
                pl.col("symbol") == symbol
            ).to_numpy().flatten()
            mean_price = sum(adj_closes) / self._window
            latest_close = view.latest_close()[symbol]
            if abs(latest_close - mean_price) > 2 * adj_closes.std():
                picks.append(symbol)

        picks = picks[:5]  # Limit to top 5
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