from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This is because low-volatility stocks are perceived to be less risky and can offer "
        "more stable returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std() / pl.col("adj_close").mean()).alias("volatility")
            )
            .collect()["volatility"]
            .to_list()
        )

        sorted_symbols = [s for _, s in sorted(zip(volatilities, view.symbols))]
        top_n_symbols = sorted_symbols[:5]

        weights = {symbol: 1.0 / len(top_n_symbols) for symbol in top_n_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest