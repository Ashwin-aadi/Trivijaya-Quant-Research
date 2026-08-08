from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualStrategy(Strategy):
    rationale = (
        "Liquidity is a proxy for market interest and can indicate the potential for more "
        "attractive trading volume. Equal weighting across highly liquid stocks can help in "
        "minimizing tracking error and reducing risk."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: list[float] = []
        symbols = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            volume_series = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()]
            if len(volume_series) < self._window:
                continue

            # Calculate the average daily trading volume as a proxy for liquidity
            avg_volume = sum(volume_series) / self._window
            liquidity_scores.append(avg_volume)
            symbols.append(symbol)

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation based on the number of selected symbols
        n_symbols = len(symbols)
        equal_weight = 1.0 / n_symbols

        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest