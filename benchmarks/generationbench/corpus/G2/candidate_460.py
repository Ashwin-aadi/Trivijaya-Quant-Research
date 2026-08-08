from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reversion strategies capitalize on the tendency of asset prices to revert "
        "to their mean levels over time. In a trailing reversion strategy, we identify "
        "overbought or oversold conditions by comparing current prices to their historical "
        "price levels from a certain lookback period."
    )

    def __init__(self, window: int = 50, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]

        # Calculate trailing average and z-score
        avg_close = (
            history.select([pl.col("adj_close").mean().alias("avg")])
            .group_by("symbol")
            .agg(pl.col("avg"))
            .to_pandas()
        )
        z_scores = (view.closes(lookback=self._window).to_polars() - avg_close)
        z_scores = (
            z_scores.join(avg_close, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") / pl.col("avg") - 1.0).alias("z_score"),
                ((pl.col("adj_close") > pl.col("avg")) & (pl.col("z_score") > self._z_score_threshold)).alias("overbought"),
            )
        )

        # Identify overbought and oversold symbols
        overbought_symbols = [s for s in z_scores.columns if "overbought_" in s]
        weights: dict[str, float] = {}
        for symbol in symbols:
            if f"overbought_{symbol}" in overbought_symbols:
                weight = 1.0 / len(overbought_symbols)
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest