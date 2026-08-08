from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relatively in recent periods to continue performing well. This strategy ranks symbols by "
        "their returns over a lookback period and invests in the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history
            .with_column((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
            .sort("session_date", descending=False)
        )

        # Group by symbol and calculate mean return over the lookback period
        grouped = (
            history
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
        )

        picks: list[str] = []
        for i in range(self._top_n):
            symbol = str(grouped.select("symbol").rows()[i][0])
            if symbol not in view.symbols:
                continue
            picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
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