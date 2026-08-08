from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that financial markets tend to move towards the average "
        "value over time. In the short term, a stock's price may deviate significantly from "
        "its long-term average. Identifying such deviations and betting on a return to the mean "
        "can generate returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(
            pl.col("adj_close").mean().alias("mean_adj_close")
        )
        z_scores = (
            closes.join(mean_close, on="session_date", how="left")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_adj_close")).abs()
                / (pl.col("adj_close") - pl.col("mean_adj_close")).rank(method="ordinal", descending=True)
                .alias("z_score")
            )
        )

        if z_scores.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if "z_score" in z_scores.columns]
        selected_symbols = [
            symbol
            for symbol in symbols
            if float(z_scores.filter(pl.col("session_date") == stamp).select(pl.col(symbol)).item()) > 2.0
        ]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in selected_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest