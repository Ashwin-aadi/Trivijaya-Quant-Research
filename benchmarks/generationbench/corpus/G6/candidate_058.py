from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityDrivenStrategy(Strategy):
    rationale = (
        "Historical data suggests that certain months exhibit higher returns due to factors such as festive seasons or financial reporting periods. This strategy aims to capture these seasonal effects by identifying stocks showing significant seasonality, especially in October and November."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(["symbol", "session_date", "close"]).pivot(
            index="session_date", columns="symbol", values="close"
        )

        seasonal_factors = {}
        for symbol in symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if symbol_history.height < self._window * 1.5:  # Allow some flexibility
                continue

            oct_nov_closes = (
                symbol_history.filter(
                    (pl.col("session_date").dt.month() == 10) | (pl.col("session_date").dt.month() == 11)
                )
                .select(["close"])
                .to_numpy()[0]
            )

            if len(oct_nov_closes) < 5:  # Need at least some data
                continue

            mean_close = pl.DataFrame(oct_nov_closes).mean().item()
            seasonal_factors[symbol] = oct_nov_closes[-1] / mean_close - 1.0

        top_symbols = sorted(seasonal_factors, key=lambda s: seasonal_factors[s], reverse=True)[:50]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest