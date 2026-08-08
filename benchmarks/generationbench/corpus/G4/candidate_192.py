from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum250d(Strategy):
    rationale = (
        "This strategy exploits the cross-sectional momentum effect by selecting stocks "
        "with positive past returns. It capitalizes on historical performance data to inform "
        "current investment decisions and aims to benefit from the persistence of momentum "
        "in stock returns."
    )

    def __init__(self, lookback: int = 250, portfolio_size: int = 30) -> None:
        self._lookback = lookback
        self._portfolio_size = portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = (
            (history["adj_close"].to_list()[1:] / history["adj_close"].shift(1).to_list() - 1.0)
            .accumulate()
            .alias("cumulative_return")
        )
        ranked_stocks = (
            history.select([pl.col("symbol"), cumulative_returns])
            .sort("cumulative_return", descending=True)
            .head(self._portfolio_size)
            .select("symbol")
            .to_dict(False)["symbol"]
        )

        if not ranked_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest