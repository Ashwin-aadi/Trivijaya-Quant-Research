from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength compared to the broader market "
        "can potentially outperform. This strategy ranks assets based on their price momentum "
        "relative to the NIFTY 100 index and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        nifty_history = history.select(pl.col("adj_close").filter(pl.col("symbol") == "NIFTY")).to_series()
        if nifty_history.is_null().sum() > 0 or len(nifty_history) < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_returns = (nifty_history / nifty_history.shift(1) - 1.0).to_list()[1:]
        asset_returns = history.select(
            [pl.col(f"adj_close/{pl.col('symbol').shift(1)}-1").alias("return") for symbol in view.symbols]
        ).transpose().drop_nulls().to_series()

        strength_scores = (asset_returns / nifty_returns).mean(axis=0)
        top_scorers = [symbol for _, symbol in sorted(zip(strength_scores.to_list(), view.symbols), reverse=True)[:self._top_n]]

        if not top_scorers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_scorers)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_scorers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest