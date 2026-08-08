from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following based on volatility scaling aims to capture trends by adjusting the "
        "threshold for trend continuation. High volatility periods suggest more caution in "
        "following trends."
    )

    def __init__(self, window: int = 20, threshold_factor: float = 1.5) -> None:
        self._window = window
        self._threshold_factor = threshold_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the volatility of the last window days
            vol = pl.DataFrame({"close": values[-self._window:]}).select(
                (pl.col("close").std() / pl.col("close").mean()).alias("volatility")
            ).item()
            
            # Calculate the trend direction
            if len(values) >= 2 * self._window:
                trend_direction = (
                    values[-self._window] - values[-2 * self._window]
                ) / (values[-1] - values[0])
                threshold = vol * self._threshold_factor

                if abs(trend_direction) > threshold:
                    trends[symbol] = 1.0

        weight = 1.0 / len(trends)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in trends.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest