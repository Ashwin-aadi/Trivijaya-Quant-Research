from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "By tilting towards low volatility, we aim to capture this excess return."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.width == 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volatility: dict[str, float] = {
            symbol: (pl.col("adj_close").std().over([pl.col("symbol"), pl.range(0, self._window)]).alias("volatility"))
            .sort(descending=False)
            .head(1)["volatility"]
            for symbol in symbols
        }
        sorted_symbols = [k for k, v in sorted(volatility.items(), key=lambda item: item[1])]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest