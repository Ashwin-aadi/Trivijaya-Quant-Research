from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue performing well. This is based on the idea that "
        "outperformance in a short period often indicates underlying strength."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes.sort("session_date")
            .shift(-self._window - 1)
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("return", descending=True)
            .select(pl.col("symbol"))
        )

        top_performers = returns.head(self._lookback)["symbol"].to_list()
        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest