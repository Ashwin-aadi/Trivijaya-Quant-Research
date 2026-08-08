from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on strong liquidity characteristics, focusing on "
        "high average daily turnover over the last 20 days. Stocks are equally weighted to ensure "
        "diversification and simplify the strategy while maintaining robust marketability and "
        "reducing execution costs."
    )

    def __init__(self, window: int = 20, min_volume: float = 1_000_000) -> None:
        self._window = window
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_symbols = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            if df.height < self._window:
                continue
            volume_mean = float(df["volume"].mean())
            if volume_mean >= self._min_volume and float(df.select(pl.last("close")).row(0)[0]) > 0:
                filtered_symbols.append(symbol)

        weights = {symbol: 1.0 / len(filtered_symbols) for symbol in filtered_symbols}
        return Signal(
            information_available_at=stamp, weights={s: weight for s in weights.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest