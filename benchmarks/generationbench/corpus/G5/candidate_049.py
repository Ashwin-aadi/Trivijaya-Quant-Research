from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that prices which are currently far from "
        "their recent average will eventually revert. This strategy aims to profit by "
        "buying underperformers and selling outperformers based on z-scores."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").mean()).alias("avg"),
                (pl.col("adj_close").std().alias("std")),
            )
            .with_columns(
                ((pl.col("adj_close") - pl.col("avg")) / pl.col("std")).alias("zscore")
            )
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_close.columns or len(avg_close[symbol]) < self._window:
                continue

            zscore = float(avg_close[f"{symbol}.zscore"][-1])
            if -2 <= zscore < -1:
                picks.append(symbol)
            elif 1 < zscore <= 2:
                picks.append(f"S{symbol}")

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.5 / len(picks)
        signal_weights = {s: weight for s in picks}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest