from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting allocates capital based on the liquidity of stocks. "
        "High-liquidity stocks are given more weight in the portfolio because they can be traded without significant impact to their price."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            volume_series = pl.Series(values[-self._window:])
            liquidity_score = volume_series.mean() / min(volume_series)
            liquidity_scores[symbol] = liquidity_score.to_numpy()[0]

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        total_lw = sum(liquidity_scores.values())
        weights = {s: l / total_lw for s, l in liquidity_scores.items()}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest