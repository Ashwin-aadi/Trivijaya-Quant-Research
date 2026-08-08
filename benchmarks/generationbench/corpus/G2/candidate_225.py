from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because investors often demand higher returns for taking on additional risk. "
        "By tilting towards low-volatility stocks, we can capture this risk premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        daily_returns = (
            (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        )
        returns_df = history.with_columns(daily_returns)
        volatilities = returns_df.groupby("symbol").agg(
            pl.col("r").std().alias("volatility")
        )

        sorted_symbols = (
            volatilities.sort(pl.col("volatility"), descending=False)["symbol"]
            .to_list()
            [: len(symbols) // 2]
        )
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest