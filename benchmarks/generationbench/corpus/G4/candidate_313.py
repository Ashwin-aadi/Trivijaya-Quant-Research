from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that have "
        "deviated significantly from their historical average prices over a 10-day period. "
        "When stock prices revert to the mean, profits are expected."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0, max_positions: int = 30) -> None:
        self._window = window
        self._threshold = threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        data = (
            history
            .select(pl.col("symbol"), pl.col("session_date"), pl.col("adj_close"))
            .filter(pl.col("symbol").is_in(symbols))
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").mean().over("symbol")) /
                pl.col("adj_close").std().over("symbol", method="quartile").alias("z_score")
            )
        )

        z_scores = data.select(pl.col("symbol"), "z_score").collect()
        extreme_z_scores = [
            (symbol, z_score)
            for symbol, z_score in zip(z_scores["symbol"], z_scores["z_score"])
            if abs(z_score) > self._threshold
        ]

        ranked_pairs = sorted(extreme_z_scores, key=lambda x: abs(x[1]), reverse=True)[:self._max_positions]
        weights = {symbol: 1.0 / len(ranked_pairs) for symbol, _ in ranked_pairs}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest