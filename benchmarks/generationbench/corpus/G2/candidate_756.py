from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum is based on the idea that securities which have performed "
        "well in the recent past are likely to continue performing well. This strategy seeks to "
        "identify such securities and overweight them."
    )

    def __init__(self, lookback_period: int = 30, n_top_securities: int = 10) -> None:
        self._lookback_period = lookback_period
        self._n_top_securities = n_top_securities

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("close") / pl.col("open").shift(self._lookback_period) - 1).alias("momentum_score"),
            )
            .sort("momentum_score", descending=True)
            .head(self._n_top_securities)
        )

        top_symbols = [row["symbol"] for row in momentum_scores.to_dicts()]
        weights = {symbol: 1.0 / self._n_top_securities for symbol in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest