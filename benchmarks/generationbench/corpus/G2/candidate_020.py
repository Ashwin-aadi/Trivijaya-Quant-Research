from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can often "
        "lead to sustained price trends. By identifying such moves, we aim to capture "
        "profitable opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols with insufficient data
        enough_history_symbols = [symbol for symbol in view.symbols if len(history[symbol].to_list()) >= self._window]

        # Calculate daily returns and volume ratio
        returns = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        volume_ratios = returns.with_columns(
            (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_ratio")
        )

        # Find symbols with significant return and higher volume
        strong_moves = (
            volume_ratios.filter(
                (pl.col("return").abs() >= self._threshold) &
                (pl.col("volume_ratio") > 1.0)
            )
            .group_by(["symbol"])
            .agg([
                pl.count().alias("count"),
                pl.mean("return").alias("mean_return"),
                pl.mean("volume_ratio").alias("mean_volume_ratio")
            ])
        )

        # Select symbols with the highest mean return and volume ratio
        top_symbols = strong_moves.sort(
            "mean_return", descending=True
        ).sort(
            "mean_volume_ratio", descending=True
        ).head(5).select(["symbol"])

        if not len(top_symbols):
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        weights_dict = {row["symbol"]: weight for row in top_symbols.iter_rows()}

        return Signal(
            information_available_at=stamp,
            weights=weights_dict
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest