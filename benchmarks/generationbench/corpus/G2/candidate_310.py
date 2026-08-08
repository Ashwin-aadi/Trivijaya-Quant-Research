from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Seasonal effects in equity markets can arise due to predictable patterns in macroeconomic "
        "events or holidays that impact investor sentiment and trading behavior. For instance, certain "
        "industries may experience higher volatility during specific times of the year."
    )

    def __init__(self, holiday_window: int = 30) -> None:
        self._holiday_window = holiday_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._holiday_window)
        if closes.height < self._holiday_window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._holiday_window:
                continue

            # Calculate the average daily return over the last holiday window days.
            returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, self._holiday_window)]
            avg_return = sum(returns) / len(returns)
            seasonality_factors[symbol] = avg_return

        # Identify symbols with above-average returns.
        above_avg_symbols = [s for s, r in seasonality_factors.items() if r > 0]
        if not above_avg_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(above_avg_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in above_avg_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest