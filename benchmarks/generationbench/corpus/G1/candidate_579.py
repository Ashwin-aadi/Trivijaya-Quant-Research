from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighting ensures that highly traded stocks receive greater allocation "
        "than less liquid ones. This strategy aims to capture the benefits of higher trading "
        "volume while spreading risk across a diversified portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close = [float(v) for v in history[symbol].to_list()[-self._window :]]
            if len(adj_close) < self._window:
                continue

            # Calculate liquidity score as the inverse of volume
            volume_series = pl.Series(adj_close).with_column(
                (1.0 / history[f"{symbol}_volume"].to_list()[-self._window :]).alias("liquidity_score")
            ).sort("session_date", descending=True).select("liquidity_score").to_list()

            liquidity_scores[symbol] = sum(volume_series) if volume_series else 0.0

        # Normalize scores to create weights
        total_score = sum(liquidity_scores.values())
        weights: dict[str, float] = {symbol: score / total_score for symbol, score in liquidity_scores.items()}

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0.0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest