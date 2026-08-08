from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High volatility can indicate heightened uncertainty or market inefficiencies, which may "
        "result in more price fluctuations. Trend following strategies aim to capture returns by "
        "identifying and exploiting trends amplified by high volatility."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            latest_close = float(view.latest_close()[symbol])
            price_changes = (
                (history.select("adj_close").filter(pl.col("symbol") == symbol)
                 .select((pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("change"))
                 .drop_nulls()
                 .sort("session_date")
                 .to_pandas()["change"])
            )
            if len(price_changes) < self._window:
                continue
            mean_change = price_changes.mean()
            std_deviation = price_changes.std(ddof=0)
            volatility_scaled_trend = (latest_close - history.select("adj_close").filter(pl.col("symbol") == symbol)
                                       .select(pl.col("adj_close")).max().to_series()).abs() / std_deviation
            if volatility_scaled_trend > self._threshold:
                symbol_data[symbol] = latest_close

        weights = {s: 1.0 / len(symbol_data) for s in symbol_data}
        return Signal(
            information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_series()[0]
    assert isinstance(newest, date)
    return newest