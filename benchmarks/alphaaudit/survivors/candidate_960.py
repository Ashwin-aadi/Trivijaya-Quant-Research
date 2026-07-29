from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the idea that assets with strong recent "
        "returns tend to continue outperforming the market. This strategy selects top performers "
        "based on their returns over a recent lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        grouped = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("close").last() / pl.col("adj_close").shift(self._window) - 1.0).alias("return"),
            )
        )

        top_performers = (
            grouped.sort(pl.col("return"), descending=True)
            .select(["symbol", "return"])
            .head(5)
            .to_dict(as_series=False)["symbol"]
        )

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest