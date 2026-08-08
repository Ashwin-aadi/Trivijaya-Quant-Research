from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed in "
        "the recent past to continue outperforming. This strategy is based on the idea that "
        "market participants tend to overreact to short-term events, leading to mean reversion "
        "in stock prices."
    )

    def __init__(self, lookback_window: int = 60, ranking_threshold: float = 0.8) -> None:
        self._lookback_window = lookback_window
        self._ranking_threshold = ranking_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        rank_scores = (
            closes.sort("session_date", descending=True)
            .select(
                pl.col(pl.Utf8).exclude("session_date").rank(method="ordinal", descending=True)
            )
            .with_column((pl.col("symbol") / pl.col("symbol").max().over("symbol")) * 100.0)
            .filter(pl.col("session_date") == stamp - date(1, 1, 1))
            .select(
                (pl.col("rank() over (partition by symbol order by session_date desc)"))
                < self._ranking_threshold
            )
            .group_by("symbol")
            .agg(
                pl.count().alias("count"),
                pl.col("session_date").min().alias("min_session_date"),
            )
        )

        symbols = rank_scores.filter(pl.col("count") == 1).select("symbol").to_series().to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest