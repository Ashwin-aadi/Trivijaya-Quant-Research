from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy focuses on selecting stocks with the lowest historical volatility to "
        "construct a low-risk portfolio. By inversely weighting the selected stocks based on "
        "their volatilities, we aim to maximize the overall stability of the portfolio."
    )

    def __init__(self, window: int = 20, bottom_percentage: float = 0.4, max_stocks: int = 50) -> None:
        self._window = window
        self._bottom_percentage = bottom_percentage
        self._max_stocks = max_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .group_by(["symbol"])
            .agg((pl.col("returns").std().alias("volatility")))
            .sort("volatility", descending=False)
        )

        # Filter to get the bottom 40% of stocks by volatility
        num_stocks = int(self._bottom_percentage * len(history))
        history = history.head(num_stocks)

        if history.height < self._max_stocks:
            return Signal(information_available_at=stamp, weights={})

        volatilities = [float(v) for v in history["volatility"].to_list()]
        symbols = [s for s in history["symbol"].to_list()]

        # Inverse weighting based on volatility
        total_volatility_inv_sum = sum(1 / v for v in volatilities)
        weights = {symbols[i]: 1.0 / (volatilities[i] * total_volatility_inv_sum) for i in range(len(symbols))}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest