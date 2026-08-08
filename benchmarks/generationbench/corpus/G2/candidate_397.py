from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion is a market phenomenon where asset prices and trading volumes return to the long-term average or mean. "
        "In short-horizon mean reversion, we expect recent deviations from the historical price range to reverse over a shorter period. "
        "This strategy aims to identify stocks that have moved away significantly from their mean price in the last 20 days and bet on them reverting."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").mean()).alias("mean"),
                ((pl.col("adj_close") - pl.col("close")).abs().mean()).alias("std"),
            )
            .collect()
        )

        if mean_close.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the z-score for each symbol
        history = (
            history.join(
                mean_close,
                on="symbol",
                how="inner",
            )
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")) / pl.col("std").alias("z_score")
            )
            .sort("session_date", descending=True)
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            z_scores = [float(v) for v in history.filter(pl.col("symbol") == symbol)["z_score"].to_list()]
            if len(z_scores) < self._window:
                continue

            latest_z_score = z_scores[-1]
            if abs(latest_z_score) > self._threshold and (
                (latest_z_score > 0 and z_scores[0] < 0)
                or (latest_z_score < 0 and z_scores[0] > 0)
            ):
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