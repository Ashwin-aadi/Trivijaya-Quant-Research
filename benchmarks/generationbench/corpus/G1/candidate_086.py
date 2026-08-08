from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows the trend of assets that have historically shown high volatility. "
        "High volatility indicates a higher probability of continuation in the current direction."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_data = history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].drop_nulls().to_list()
            volatility = _calculate_volatility(close_data)
            volatilities[symbol] = volatility

        top_symbols = [
            s for (s, v) in sorted(volatilities.items(), key=lambda item: -item[1])
        ][: self._threshold]

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


def _calculate_volatility(data: list[float]) -> float:
    mean_return = sum(data) / len(data)
    variance = sum((x - mean_return) ** 2 for x in data) / (len(data) - 1)
    volatility = (variance**0.5) * (252**0.5)
    return volatility