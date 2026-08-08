from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion strategies aim to profit from the tendency of prices to return to their "
        "mean over a given period. This strategy identifies stocks that have deviated significantly "
        "from their mean price and bets on a reversal towards the mean."
    )

    def __init__(self, window: int = 5, deviation_threshold: float = 0.1) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_price"))
        )
        
        recent_closes = view.closes(lookback=None)
        symbols_with_mean = set(mean_prices["symbol"])
        symbols_to_consider = [s for s in view.symbols if s in symbols_with_mean]
        
        picks: list[str] = [
            symbol
            for symbol in symbols_to_consider
            if (recent_closes[symbol].to_list()[-1] - mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item()) / mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item() < -self._deviation_threshold or 
               (mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item() - recent_closes[symbol].to_list()[-1]) / mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item() > self._deviation_threshold
        ]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest