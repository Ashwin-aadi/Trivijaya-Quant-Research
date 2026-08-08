from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have more stable returns and are often less affected by market volatility. "
        "By tilting the portfolio towards low-volatility stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        volatilities = {symbol: 0.0 for symbol in symbols}

        for _, row in history.filter(pl.col("session_date") != view.as_of).iter_rows():
            if any(pd.isnull(row) for row in row):
                continue

            adj_close_series = [float(val) for val in row[1:]]

            daily_returns = [
                (adj_close_series[i] / adj_close_series[i - 1] - 1.0)
                for i in range(1, len(adj_close_series))
            ]

            if all(r == 0.0 for r in daily_returns):
                continue

            volatility = (sum([r**2 for r in daily_returns]) / self._window) ** 0.5
            volatilities[row[0]] += volatility

        mean_volatility = sum(volatilities.values()) / len(symbols)
        low_vol_symbols = [
            symbol for symbol, v in volatilities.items() if v <= mean_volatility
        ]

        weight = 1.0 / len(low_vol_symbols) if low_vol_symbols else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())
    assert isinstance(newest[0], date)
    return newest[0]