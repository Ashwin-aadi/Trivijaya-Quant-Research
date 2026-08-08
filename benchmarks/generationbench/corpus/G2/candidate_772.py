from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price levels revert to a mean over time. By identifying symbols that have deviated "
        "significantly from their trailing average, we can anticipate a reversion to the mean."
    )

    def __init__(self, window: int = 60, mean_window: int = 30) -> None:
        self._window = window
        self._mean_window = mean_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        trailing_mean = (
            view.history(lookback=self._mean_window)
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("trailing_mean"))
            .to_dict(as_series=False)
        )
        
        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in trailing_mean or symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            latest_close = float(view.latest_close()[symbol])
            mean_close = trailing_mean[symbol]["trailing_mean"]
            
            # Calculate the z-score as a measure of deviation from the trailing mean
            score = (latest_close - mean_close) / mean_close if mean_close != 0 else 0.0
            
            reversion_scores[symbol] = score

        top_symbols: list[str] = sorted(reversion_scores, key=reversion_scores.get, reverse=True)[:5]
        
        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest