from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "During periods of range compression, asset prices oscillate within a narrow band "
        "indicating low volatility and reduced directional momentum. This can signal an impending breakout, "
        "which can lead to higher returns when entering positions on the breakout."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = pl.DataFrame({symbol: history[symbol]["adj_close"].to_list()[-1] for symbol in symbols}).transpose()
        min_prices = history.group_by("symbol").agg(pl.col("adj_close").min()).select(["symbol", "adj_close"]).rename({"adj_close": f"min_{self._window}_day"})
        max_prices = history.group_by("symbol").agg(pl.col("adj_close").max()).select(["symbol", "adj_close"]).rename({"adj_close": f"max_{self._window}_day"})

        recent_min = min_prices.join(recent_closes, on="symbol", how="left").with_columns((pl.col(f"adj_close") - pl.col(f"min_{self._window}_day")).alias("recent_range"))
        recent_max = max_prices.join(recent_closes, on="symbol", how="left").with_columns((pl.col(f"adj_close") - pl.col(f"max_{self._window}_day")).alias("recent_range"))

        range_compression_ratio = (recent_min["recent_range"] / recent_max["recent_range"]).to_list()
        sorted_indices = [i for _, i in sorted(zip(range_compression_ratio, symbols), reverse=True)]

        top_symbols = sorted_indices[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest