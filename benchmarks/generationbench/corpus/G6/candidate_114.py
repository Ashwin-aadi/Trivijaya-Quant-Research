from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with strong relative performance against the broader market "
        "by using a 60-day lookback period. It selects top performers weekly and exits if they underperform or are held for over 6 months."
    )

    def __init__(self, window: int = 60, top_n: int = 10, hold_period: int = 180) -> None:
        self._window = window
        self._top_n = top_n
        self._hold_period = hold_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes().select([pl.col(col).last() for col in view.symbols])
        market_close = view.closes(lookback=self._window)["NIFTY"].last()
        relative_strengths = (
            (closes / market_close - 1.0) * 100
        ).to_dict(as_series=False)

        sorted_symbols = [
            k for _, v in sorted(relative_strengths.items(), key=lambda item: item[1], reverse=True)
            if v > 2 and history.select(pl.col("symbol") == k).height < self._hold_period
        ][:self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest