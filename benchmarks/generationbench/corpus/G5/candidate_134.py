from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves identifying trends by comparing the "
        "recent price change to a volatility measure. Higher than average positive changes "
        "are considered buys, while higher negative changes are sells."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        daily_changes = {symbol: (latest_closes[symbol] - float(history.select(pl.col(symbol)).item())) for symbol in view.symbols}
        
        volatility = history.groupby().agg(
            pl.col("adj_close").std()
        ).select(
            pl.col("adj_close").alias("volatility")
        ).collect()[0]["volatility"]

        buys, sells = [], []
        for symbol, change in daily_changes.items():
            if change > 0:
                if change / volatility >= self._threshold:
                    buys.append(symbol)
            elif change < 0:
                if -change / volatility >= self._threshold:
                    sells.append(symbol)

        weights = {s: 1.0 / len(buys) for s in buys} if buys else {}
        weights.update({s: -1.0 / len(sells) for s in sells}) if sells else {}

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest