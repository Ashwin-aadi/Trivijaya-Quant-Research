from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well recently to continue performing well. This strategy buys top performers and "
        "sells bottom performers based on recent returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        returns = (
            history
            .with_columns(
                (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        if returns.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_returns = (
            returns
            .sort("avg_return", descending=True)
            .select(["symbol", "avg_return"])
            .to_pandas()
        )
        
        top_symbols = sorted_returns["symbol"].tolist()[: self._top_n]
        bottom_symbols = sorted_returns["symbol"].tolist()[-self._top_n:]

        weight_top = 1.0 / len(top_symbols)
        weight_bottom = -1.0 / len(bottom_symbols)

        weights = {s: weight_top for s in top_symbols}
        for s in bottom_symbols:
            weights[s] = weight_bottom

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest