from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalMomentum(Strategy):
    rationale = (
        "This strategy leverages historical patterns related to economic events and seasonal "
        "variations in stock performance. It aims to capture upward momentum after significant "
        "seasonal events like Diwali, ensuring alignment with historical seasonality patterns."
    )

    def __init__(self, window: int = 50, lookback_days: int = 90) -> None:
        self._window = window
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        diwali_date = date(year=history.height - 1, month=10, day=25)
        current_session_date = stamp
        if (current_session_date - diwali_date).days < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_to_consider = [s for s in view.symbols if f"_{s}" in history.columns]
        if not symbols_to_consider:
            return Signal(information_available_at=stamp, weights={})

        price_changes = (
            history.filter(
                pl.col("session_date") > diwali_date
            )  # Filter sessions after Diwali
            .select(pl.col(symbols_to_consider))
            .with_column((pl.col(symbols_to_consider) - pl.col(symbols_to_consider).shift(self._window)) / pl.col(symbols_to_consider).shift(self._window) * 100)
            .sort("session_date", descending=False)
            .head(30)
        )

        selected_symbols = [symbol for symbol in symbols_to_consider if symbol in price_changes.columns]
        weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest