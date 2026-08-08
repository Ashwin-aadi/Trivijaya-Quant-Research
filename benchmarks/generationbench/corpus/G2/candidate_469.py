from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies exploit deviations from the mean price. In a short horizon, "
        "prices that have deviated significantly from their historical average are likely to "
        "reverse and return towards the mean."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean())
            .to_series()
            .item()
        )
        closes = view.closes(lookback=self._window)

        # Filter out symbols that do not have enough data
        if any(closes.height < self._window):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the z-score for each symbol
        z_scores = (
            (closes - mean_close).select(
                pl.arange(1, closes.height + 1).zip_with(pl.col("adj_close"), lambda i, v: (v - mean_close[i]) / history.select(pl.col("adj_close").std()).to_series().item())
            )
        ).to_dict(True)

        # Select symbols with the most negative z-scores
        candidates = [
            symbol for symbol, score in z_scores.items() if float(score[-1]) <= -2.0
        ][:5]

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest