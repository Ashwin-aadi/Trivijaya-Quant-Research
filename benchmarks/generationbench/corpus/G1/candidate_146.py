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

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
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

            close_mean = sum(closes[-self._window:]) / self._window
            open_mean = sum(opens[-self._window:]) / self._window

            daily_returns = [(c - o) / o for c, o in zip(closes, opens)]
            volatility = pl.DataFrame({"daily_return": daily_returns}).select(
                (pl.col("daily_return") ** 2).mean().alias("volatility")
            ).row(0)[0] ** 0.5

            trend_threshold = open_mean + self._threshold * volatility
            if closes[-1] > trend_threshold:
                symbol_data[symbol] = {"open": opens, "close": closes}

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        total_value = sum(len(v["close"]) for v in symbol_data.values())
        weight_per_symbol = 1.0 / len(symbol_data)

        weights: dict[str, float] = {}
        for symbol, data in symbol_data.items():
            close_prices = [float(v) for v in data["close"].drop_nulls().to_list()]
            weight = (len(close_prices) - 1) * weight_per_symbol
            weights[symbol] = weight

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest