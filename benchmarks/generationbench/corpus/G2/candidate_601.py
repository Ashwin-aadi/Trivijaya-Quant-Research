from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionShortHorizon(Strategy):
    rationale = (
        "Mean reversion strategies profit from mean-reverting markets. The idea is that "
        "prices tend to move back towards an average value over time. In a short horizon "
        "mean reversion strategy, we identify stocks that have deviated significantly "
        "from their moving average and expect them to revert."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        mean_close = (
            history.select(pl.col("adj_close").mean().alias("ma"))
            .to_vertical()
            .select(pl.col("adj_close") / pl.col("ma") - 1.0)
        )
        
        filtered_symbols = []
        for symbol in symbols:
            ratio = float(mean_close[symbol][mean_close[symbol].height - 1])
            if abs(ratio) > 0.2:  # Threshold for deviation
                filtered_symbols.append(symbol)

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest