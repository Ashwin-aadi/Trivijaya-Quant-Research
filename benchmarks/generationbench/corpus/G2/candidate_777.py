from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Lower-volatility stocks tend to outperform higher-volatility ones over long periods. "
        "This is due to a combination of lower risk and the market's tendency to overreact to volatility. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture these persistent returns."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        history = history.select(
            pl.col("session_date").alias("date"),
            *[pl.col(symbol) for symbol in symbols],
        )
        adj_closes = history[symbols].to_numpy().T

        volatilities = [
            (np.std(adj_closes[:, i]) if np.any(np.isfinite(adj_closes[:, i])) else 0)
            for i in range(len(symbols))
        ]
        sorted_indices = np.argsort(volatilities)

        weights: dict[str, float] = {}
        for index in sorted_indices:
            symbol = symbols[index]
            weight = (1.0 - index / len(symbols)) * (1.0 / len(symbols))
            if weight > 0:
                weights[symbol] = weight

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