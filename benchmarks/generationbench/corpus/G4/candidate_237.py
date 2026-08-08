from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed higher-volatility counterparts due to "
        "reduced risk and the persistent low-volatility factor in equity markets. This strategy aims to "
        "capitalize on this phenomenon by selecting a portfolio heavily weighted towards low-volatility "
        "stocks."
    )

    def __init__(self, window: int = 252, min_symbols: int = 30, max_symbols: int = 50) -> None:
        self._window = window
        self._min_symbols = min_symbols
        self._max_symbols = max_symbols

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .sort("volatility", descending=False)
        )

        if symbol_returns.height < self._min_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in symbol_returns.to_dicts()[:self._max_symbols]]

        volatility_weights = {
            symbol: 1.0 / len(top_symbols) * (2 if i == 0 else 1)
            for i, symbol in enumerate(top_symbols)
        }

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in volatility_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest