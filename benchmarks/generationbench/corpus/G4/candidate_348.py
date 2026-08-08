from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy leverages cross-sectional momentum in the Indian equity market by "
        "identifying stocks that have outperformed or underperformed relative to their historical "
        "performance. It aims to capture the persistency of past performance in stock returns."
    )

    def __init__(self, lookback_days: int = 180, top_n: int = 20) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        # Calculate momentum score
        def calc_momentum(row: pl.Series) -> float:
            close_now = row[-1]
            close_end_lookback = row[0]
            return ((close_now - close_end_lookback) / close_end_lookback) * 100

        history = (
            history.lazy()
            .group_by("symbol")
            .with_column(
                (pl.col("adj_close").shift(-self._lookback_days).alias("close_end_lookback"))
            )
            .select(pl.all().except_("session_date"))
            .rows_as_dict()
            .map(calc_momentum)
        )

        # Sort by momentum score
        history = pl.DataFrame(history)
        top_decile, bottom_decile = self._get_top_bottom_deciles(history)

        weights = {symbol: 1.0 / len(top_decile) for symbol in top_decile}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )

    def _get_top_bottom_deciles(self, momentum_scores: pl.DataFrame) -> tuple[list[str], list[str]]:
        sorted_symbols = (
            momentum_scores.sort("momentum_score", descending=True)
            .select(["symbol"])
            .to_series()
            .to_list()
        )
        top_n = self._top_n
        bottom_n = len(sorted_symbols) - top_n
        return sorted_symbols[:top_n], sorted_symbols[-bottom_n:]


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest