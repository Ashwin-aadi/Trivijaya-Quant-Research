from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market stability and reduced volatility. "
        "This strategy aims to identify stocks where the recent price range has compressed, "
        "suggesting a potential for breakout or consolidation in a favorable direction."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed_stocks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            open_prices = [float(v) for v in df["open"].to_list()]
            close_prices = [float(v) for v in df["close"].to_list()]

            min_price = min(open_prices + close_prices)
            max_price = max(open_prices + close_prices)

            price_range = max_price - min_price
            if price_range < 0.1 * (max_price + min_price):  # Considering a small range as compressed
                compressed_stocks.append(symbol)

        if not compressed_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().item()
    assert isinstance(newest, date)
    return newest