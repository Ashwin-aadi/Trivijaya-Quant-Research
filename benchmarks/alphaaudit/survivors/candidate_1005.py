from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and may have better risk-adjusted returns. "
        "By tilting the portfolio towards low-volatility stocks, we aim to reduce overall portfolio risk while potentially enhancing return."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("volatility"))
            .sort("volatility", descending=False)
        )

        if volatility.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in volatility.to_dicts()[:self._window]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest