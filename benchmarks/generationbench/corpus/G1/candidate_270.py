from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identify stocks with the highest relative strength against the NIFTY 100 index "
        "to capitalize on strong performance within the broader market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        nifty_close = closes.filter(pl.col("symbol") == "NIFTY 100").select(
            pl.col("session_date"), pl.col("adj_close").alias("nifty_close")
        )

        if nifty_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol == "NIFTY 100":
                continue

            close_values = [float(v) for v in closes.filter(pl.col("symbol") == symbol).select(
                pl.col("adj_close")
            ).column("adj_close").to_list()]

            nifty_values = [float(v) for v in nifty_close.select("nifty_close").column("nifty_close").to_list()]

            if len(close_values) < self._window or len(nifty_values) < self._window:
                continue

            close_ratio = sum([close / nifty if nifty != 0 else float('inf') for close, nifty in zip(close_values, nifty_values)]) / min(len(close_values), self._window)

            relative_strengths[symbol] = close_ratio

        sorted_stocks = [
            stock
            for _, stock in sorted(relative_strengths.items(), key=lambda item: -item[1])
        ][:5]

        if not sorted_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest