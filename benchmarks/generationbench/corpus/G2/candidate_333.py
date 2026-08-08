from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed high-volatility stocks. This "
        "effect can be attributed to the market's tendency to overreact to volatility and "
        "the risk premium that investors demand for holding risky assets."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").std().alias("volatility")))
            .sort("volatility", descending=False)["volatility"]
            .to_list()
        )

        if len(volatilities) < self._window:
            return Signal(information_available_at=stamp, weights={})

        rank = [i for i, v in enumerate(volatilities)]
        top_symbols = [view.symbols[i] for i in sorted(rank)[:5]]
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