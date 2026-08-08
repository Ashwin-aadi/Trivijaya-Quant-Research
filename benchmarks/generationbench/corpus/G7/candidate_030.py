from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks with the highest relative strength against the broad market "
        "index. Relative strength is calculated by comparing a stock's closing price over the last 30 days "
        "to the average closing price of all NIFTY 100 constituents during the same period."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_avg_closes = history.select(
            pl.col("adj_close").mean().alias("avg_close")
        )
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = (
            history.filter(pl.col("symbol").is_in(symbols))
            .group_by("session_date", "symbol")
            .agg(
                (pl.col("adj_close") / nifty_avg_closes["avg_close"] * 100).alias("rs")
            )
            .sort("rs", descending=True)
        ).select("rs")

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_10 = [row[0] for row in closes.head(10)["rs"].to_list()]
        weight = 1.0 / len(top_10)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_10},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest