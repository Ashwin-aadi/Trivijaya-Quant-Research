from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices which have deviated significantly from their "
        "historical mean will tend to return to it over time. In the short term, we can use recent "
        "price action to identify such deviations and generate trading signals."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = closes.columns
        mean_close = (
            history.select(pl.col("symbol").alias("symbol"), pl.col("adj_close"))
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean_close"))
        )

        deviation_scores: list[float] = []
        for symbol in symbols:
            if symbol not in mean_close.columns:
                continue
            symbol_closes = closes[symbol]
            recent_mean = float(mean_close.filter(pl.col("symbol") == symbol)["mean_close"].item())
            recent_closes = [float(v) for v in symbol_closes.drop_nulls().to_list()]
            deviation_score = sum(abs(c - recent_mean) for c in recent_closes[-self._window:])
            deviation_scores.append(deviation_score)

        top_symbols: list[str] = [
            symbol
            for _, (symbol, _) in sorted(
                zip(symbols, deviation_scores), key=lambda x: x[1], reverse=True
            )[: self._top_n]
        ]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest