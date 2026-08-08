from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price levels that revert to the mean over time can be exploited for profitable "
        "trading. By identifying symbols that have deviated significantly from their "
        "average price level and are showing signs of reversion, we aim to capitalize on "
        "this tendency."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close_by_symbol = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_adj_close"),
                (pl.col("adj_close") - pl.col("adj_close").mean()).abs()
                / pl.col("adj_close").std()
                .over("symbol")
                .alias("z_score"),
            )
        ).collect()

        avg_close_by_symbol = (
            avg_close_by_symbol
            .filter(avg_close_by_symbol["z_score"] > self._z_score_threshold)
            .sort(
                "z_score",
                descending=False,
            )
            .select(["symbol", "avg_adj_close"])
        )

        if avg_close_by_symbol.height < 1:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in avg_close_by_symbol.to_dicts()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight
                for s in picks
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest