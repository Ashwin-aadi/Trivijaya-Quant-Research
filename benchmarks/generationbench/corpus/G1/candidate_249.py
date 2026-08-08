from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time. "
        "By tilting our portfolio towards low volatility, we can potentially enhance returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_vols = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").std().alias("volatility")))
            .sort("volatility", descending=False)["volatility"]
            .to_list()
        )

        if len(symbol_vols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s, _ in zip(view.symbols, symbol_vols)[:5]]
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