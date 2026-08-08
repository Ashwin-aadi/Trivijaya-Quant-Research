from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following identifies trends by comparing the current price "
        "movement to its historical volatility. A breakout above a threshold multiple of the "
        "volatility indicates a strong upward trend, prompting entry into the market."
    )

    def __init__(self, window: int = 25, threshold: float = 1.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 2 + 1)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        symbol_data: dict[str, pl.DataFrame] = {}

        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            if df.height < self._window * 2 + 1:
                continue

            opens = [float(v) for v in df["open"].drop_nulls().to_list()]
            closes = [float(v) for v in df["close"].drop_nulls().to_list()]

            price_changes = [c / o - 1.0 for c, o in zip(closes[1:], opens)]
            mean_change = sum(price_changes) / len(price_changes)
            std_deviation = (sum((x - mean_change) ** 2 for x in price_changes) / len(
                price_changes)) ** 0.5

            if abs(mean_change) > self._threshold * std_deviation:
                symbol_data[symbol] = df.tail(self._window)

        picks: list[str] = [symbol for symbol, data in symbol_data.items()
                            if not data.is_empty()]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest