from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Companies that maintain higher relative strength compared to their peers are "
        "potentially outperforming the broader market. This can be due to better business "
        "performance or favorable market conditions. Identifying such companies early "
        "can provide an advantage in the long run."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window).with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window - 1) - 1.0).alias("rs")
        )
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_rs = (
            closes.group_by("symbol").agg(pl.col("rs").mean().alias("avg_rs"))
        ).to_dict(as_series=False)
        top_symbols = sorted(avg_rs.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest