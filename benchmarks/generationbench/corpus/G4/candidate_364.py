from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion against a trailing reference level aims to capitalize on the tendency "
        "for asset prices to return to their historical averages. Significant deviations from "
        "these levels can present buying or selling opportunities."
    )

    def __init__(self, window: int = 50, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_deviation_map: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_series = history.filter(pl.col("symbol") == symbol).select(
                pl.col("adj_close")
            )
            tma = close_series.select(
                (pl.col("adj_close").mean()).alias("tma")
            ).collect().rows()[0][0]
            current_close = float(view.latest_close()[symbol])
            deviation = abs((current_close - tma) / tma) * 100
            if deviation > self._threshold:
                symbol_deviation_map[symbol] = deviation

        symbols_with_high_deviations: list[str] = [
            k for k, v in sorted(symbol_deviation_map.items(), key=lambda item: item[1], reverse=True)
        ][:20]
        weight_per_symbol = 1.0 / len(symbols_with_high_deviations) if symbols_with_high_deviations else 0.0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight_per_symbol for symbol in symbols_with_high_deviations},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest