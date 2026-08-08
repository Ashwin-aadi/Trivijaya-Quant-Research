from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices which have deviated significantly from their "
        "historical average will eventually revert to it. In the context of NIFTY 100 constituents, "
        "a stock whose price has dropped sharply relative to its mean over a short period is likely "
        "to return towards this mean in the near future."
    )

    def __init__(self, window: int = 10, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("symbol")).pivot(index="session_date", columns="symbol").fill_none("zero")
        means = closes.mean(axis=0).select(pl.all().mean())
        z_scores = (closes - means) / closes.std(axis=0)
        recent_closes = view.closes(lookback=self._window)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in z_scores.columns or symbol not in recent_closes.columns:
                continue
            z_score = float(z_scores[symbol].max())
            close = float(recent_closes[symbol][0])
            if abs(z_score) >= self._threshold and (close - means[symbol]) < 0:
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest