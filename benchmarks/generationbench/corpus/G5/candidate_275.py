from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy combines two characteristics: a 20-day closing price above the 50-day simple moving average (SMA) and "
        "a high volume on the most recent day. These signals are weak individually but combined they suggest increased "
        "momentum and confidence from market participants."
    )

    def __init__(self, window_20: int = 20, window_50: int = 50) -> None:
        self._window_20 = window_20
        self._window_50 = window_50

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_20)
        history = view.history(lookback=max(self._window_20, self._window_50))
        if closes.height < self._window_20 or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_50 = _sma(view.history(), window=self._window_50)
        merged = closes.join(sma_50, on="symbol", how="inner")

        if (merged["adj_close"] > merged["sma_50"]) & (
            merged["volume"] >= merged["volume"].quantile(0.75)
        ):
            picks = [
                symbol
                for symbol in view.symbols
                if symbol in merged.select("symbol").to_list()
            ]
            weight = 1.0 / len(picks)
            return Signal(
                information_available_at=stamp, weights={s: weight for s in picks}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _sma(df: pl.DataFrame, window: int) -> pl.DataFrame:
    sma = df.group_by("symbol").agg(
        (pl.col("adj_close") / window).sum().alias(f"sma_{window}")
    )
    return sma