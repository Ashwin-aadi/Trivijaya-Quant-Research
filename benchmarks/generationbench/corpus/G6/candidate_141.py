from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy focuses on selecting stocks with lower historical volatility using a 20-day standard deviation of daily returns. "
        "By tilting the portfolio towards low-volatility stocks and rebalancing monthly, we aim to reduce overall risk and enhance diversification."
    )

    def __init__(self, window: int = 20, top_n: int = 30, loss_bound: float = -15.0) -> None:
        self._window = window
        self._top_n = top_n
        self._loss_bound = loss_bound

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").std().alias("volatility"))
            .collect()
        )

        # Filter out symbols without enough data
        valid_symbols = [s for s in view.symbols if s in returns.columns]

        if len(valid_symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Sort by volatility and select bottom 30%
        selected_symbols = [
            s for _, s in sorted(zip(returns[valid_symbols].select("volatility").to_numpy().flatten(), valid_symbols))[: self._top_n]
        ]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest