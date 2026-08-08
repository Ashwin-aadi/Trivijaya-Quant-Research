from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stock market phenomena exhibit seasonal patterns. For example, commodity prices "
        "and consumer staples often show higher returns during specific months of the year due to "
        "weather conditions or holiday effects. By exploiting these regularities, we can generate "
        "predictive signals for trading."
    )

    def __init__(self, window: int = 30, seasonality_lookback: int = 12) -> None:
        self._window = window
        self._seasonality_lookback = seasonality_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the average close over the lookback period
            avg_close = sum(values[-self._seasonality_lookback - 1:-1]) / (
                self._seasonality_lookback
            )

            # Compute the seasonality factor for this symbol
            recent_close = values[-1]
            seasonality_factor = (recent_close - avg_close) / avg_close

            if seasonality_factor > 0:
                seasonality_factors[symbol] = seasonality_factor

        if not seasonality_factors:
            return Signal(information_available_at=stamp, weights={})

        # Normalize the factors to sum up to 1
        total_factor = sum(seasonality_factors.values())
        normalized_factors = {s: f / total_factor for s, f in seasonality_factors.items()}

        weight = 1.0 - min(normalized_factors.values())  # Allocate more weight to symbols with higher factor

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in normalized_factors.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest