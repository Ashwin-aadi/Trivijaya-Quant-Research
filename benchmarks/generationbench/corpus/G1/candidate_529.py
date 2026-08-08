from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion2d(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices and profits tend to return to the long-term "
        "mean. If a security has deviated significantly from its mean, it is likely to revert."
    )

    def __init__(self, window: int = 2, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            mean_price = sum(prices) / len(prices)
            z_score = (prices[-1] - mean_price) / pl.col("adj_close").std().over(prices).alias("z_score")
            if abs(z_score) >= self._z_score_threshold:
                mean_reversion_scores[symbol] = z_score

        picks = sorted(mean_reversion_scores, key=mean_reversion_scores.get, reverse=True)[:2]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest