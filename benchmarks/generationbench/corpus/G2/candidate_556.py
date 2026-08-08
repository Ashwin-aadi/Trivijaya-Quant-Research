from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price levels that revert to their mean over time can be exploited by buying "
        "underperformers and selling outperformers. This strategy aims to capture the reversionary "
        "momentum in a crowded market."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history.columns) < 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate the simple moving average of close prices
        sma_close = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window=self._window).alias(f"sma_{self._window}"))
            )
            .select(["symbol", "session_date", f"sma_{self._window}"])
            .drop_nulls()
        )

        # Calculate the z-score for each symbol
        z_scores = (
            closes.join(sma_close, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") / pl.col(f"sma_{self._window}") - 1.0).alias("z_score")
            )
            .select(["symbol", "session_date", "z_score"])
        )

        # Get the latest z-scores
        latest_z_scores = _latest_z_scores(z_scores)

        # Identify outperformers and underperformers based on their z-scores
        outperformers: list[str] = []
        underperformers: list[str] = []
        for symbol, z_score in latest_z_scores.items():
            if z_score > 0.5:
                outperformers.append(symbol)
            elif z_score < -0.5:
                underperformers.append(symbol)

        # Determine weights based on the reversionary signal
        weight_out = 1.0 / len(outperformers) if outperformers else 0
        weight_under = 1.0 / len(underperformers) if underperformers else 0

        return Signal(
            information_available_at=stamp,
            weights={
                s: weight_out for s in outperformers
            } | {s: -weight_under for s in underperformers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _latest_z_scores(z_scores: pl.DataFrame) -> dict[str, float]:
    latest_closes = z_scores.sort("session_date", descending=True).select(["symbol", "z_score"])
    return {row[0]: row[1] for row in latest_closes.iter_rows()}