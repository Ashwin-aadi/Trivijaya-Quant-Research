from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects suggest that certain times of the year have historically "
        "exhibited stronger returns. This strategy aims to identify and capitalize on such "
        "patterns."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)  # Assuming 252 trading days in a year
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        seasonality_factors: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes:
                continue

            values = [float(v) for v in closes]
            mean_return = sum(values) / len(values)
            seasonality_factors[symbol] = 0.0
            current_year_close = view.closes().get_column(symbol).to_list()[-1]

            # Calculate seasonal factor based on the difference between yearly closes and mean return
            for i, close in enumerate(values):
                if (i + 1) % 252 == 0:  # Assuming one year of data every 252 days
                    seasonality_factors[symbol] += (close - mean_return)**2

        # Normalize factors to sum up to the total portfolio value
        total_seasonal_factor = sum(seasonality_factors.values())
        seasonal_signals = {s: f / total_seasonal_factor for s, f in seasonality_factors.items()}

        if not seasonal_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(view.symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in seasonal_signals.items() if w > 0} or {s: weight for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest