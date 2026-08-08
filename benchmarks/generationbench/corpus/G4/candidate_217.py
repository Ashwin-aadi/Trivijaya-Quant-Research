from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "The strategy exploits the tendency for stock prices to revert to their historical "
        "average levels over a 50-day period. Extreme deviations from this mean are used as "
        "trading signals for short-term profitable trades."
    )

    def __init__(self, lookback: int = 50, threshold: float = 2.0) -> None:
        self._lookback = lookback
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        history = (
            history
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("mean"),
                (pl.col("adj_close") - pl.col("adj_close").mean()).std().alias("std_dev"),
            )
        )

        z_scores = (
            view.closes(lookback=self._lookback)
            .join(history, on="symbol", how="semi")
            .with_columns(
                ((pl.col("adj_close") - pl.col("mean")) / pl.col("std_dev")).alias("z_score"),
            )
        )

        if z_scores.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        top_negatives = [symbol for symbol, score in z_scores.sort("z_score", descending=False).select(["symbol"]).to_numpy().tolist()[:20]]
        bottom_positives = [symbol for symbol, score in z_scores.sort("z_score").select(["symbol"]).to_numpy().tolist()[-20:]]

        if not top_negatives or not bottom_positives:
            return Signal(information_available_at=stamp, weights={})

        weight_negatives = 1.0 / len(top_negatives)
        weight_positives = 1.0 / len(bottom_positives)

        weights = {symbol: -weight_negatives for symbol in top_negatives}
        for symbol in bottom_positives:
            weights[symbol] = weight_positives

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest