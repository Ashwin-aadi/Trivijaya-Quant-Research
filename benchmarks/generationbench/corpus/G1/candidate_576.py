from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that assets with the best recent performance "
        "are likely to continue performing well in the near future. This strategy buys "
        "the top performers and sells or holds others."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not closes.columns:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                pl.col("adj_close").sort(descending=True).head(1).alias("max_adj_close"),
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window - 1) - 1.0)
                .mean()
                .alias("avg_return"),
            )
            .collect()
        )

        ranked = symbol_data.with_columns(
            (
                pl.col("avg_return")
                .rank(method="ordinal", descending=True)
                .alias("rank")
            ).filter(pl.col("rank") <= self._top_n)
        )

        picks = [row["symbol"] for row in ranked.iter_rows() if "rank" in row]
        weight = 1.0 / len(picks) if picks else 0

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