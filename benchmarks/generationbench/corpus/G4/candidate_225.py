from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This strategy aims to capitalize on this empirical observation by tilting the portfolio towards low-volatility stocks."
    )

    def __init__(self, window: int = 60, portfolio_size: int = 25) -> None:
        self._window = window
        self._portfolio_size = portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        stock_data = (
            history.group_by("symbol")
                   .agg(
                       (pl.col("adj_close").std().alias("volatility")),
                   )
        )

        top_stocks = (
            stock_data.sort("volatility", descending=False)
                      .select(["symbol"])
                      .head(self._portfolio_size)
                      .to_dict(as_series=False)["symbol"]
        )

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest