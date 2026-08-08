from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "High liquidity stocks tend to be more efficient in pricing and less prone to trading "
        "anomalies. By equal-weighting these stocks, we aim to capture the benefits of market "
        "efficiency while maintaining a balanced portfolio."
    )

    def __init__(self, lookback_window: int = 20) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume = history.select(pl.col("symbol"), pl.col("volume"))
        total_volume = volume.group_by("symbol").agg(volume.sum().alias("total_volume"))

        # Normalize volumes by their sum
        normalized_volumes = (
            total_volume.join(history, on="symbol", how="inner")
                         .with_columns((pl.col("volume") / pl.col("total_volume")).alias("normalized_volume"))
        )

        # Rank symbols by normalized volume
        ranked_symbols = normalized_volumes.sort("normalized_volume", descending=True).select(pl.col("symbol"))

        top_symbols = ranked_symbols.head(5)  # Adjust the number of top symbols as needed

        if not top_symbols.height:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        signal_weights = {row["symbol"]: weight for row in top_symbols.iter_rows()}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest