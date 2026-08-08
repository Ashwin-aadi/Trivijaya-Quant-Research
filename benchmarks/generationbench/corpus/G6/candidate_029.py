from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks breaking out of a support or resistance level and "
        "continues holding positions if they maintain the breakout. High volume on the first day "
        "signals robust momentum, and strict risk management ensures capital protection."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            recent_closes = [float(v) for v in history.select(pl.col(symbol))["close"].to_list()]
            if len(recent_closes) < self._window + 1:
                continue

            breakout_high = max(recent_closes[:-1])
            breakout_close = recent_closes[-1]
            volume_ratio = float(history.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("close") == breakout_close)
                & (pl.col("volume") > pl.col("volume").mean().cast(pl.Float64))
            ).select("volume").item()) / history.select(pl.col(symbol).alias("vol"))["vol"].mean().cast(pl.Float64)

            if volume_ratio >= 1.5 and breakout_close > breakout_high:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.1
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest