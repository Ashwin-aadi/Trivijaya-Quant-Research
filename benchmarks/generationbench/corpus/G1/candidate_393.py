from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are less sensitive to market-wide volatility and "
        "are often associated with higher risk-adjusted returns. By tilting towards these "
        "stocks, we aim to reduce overall portfolio risk while potentially enhancing returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            daily_returns = [
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, self._window)
            ]
            volatility = pl.DataFrame({"returns": daily_returns}).select(
                (pl.col("returns").std() * 252**0.5).alias("volatility")
            ).item()
            volatilities.append(volatility)

        min_volatility = min(volatilities)
        picks: list[str] = [symbol for symbol, vol in zip(view.symbols, volatilities) if vol == min_volatility]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest