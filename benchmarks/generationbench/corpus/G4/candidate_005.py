from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies breakout continuation patterns in the Indian market "
        "by looking for significant price movements and volume increases. It aims to capture"
        "sustained trends by entering positions on pullbacks or continuations of strong moves."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_threshold = 0.03
        volume_multiplier = 2.0

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol)
            if hist.is_empty():
                continue
            recent_closes = hist.select("adj_close").to_series().to_list()
            high_20d, low_20d = max(recent_closes), min(recent_closes)
            today_high, today_low = float(hist.select("high").to_series()[0]), float(
                hist.select("low").to_series()[0]
            )
            adj_close_today = float(hist.select("adj_close").to_series()[0])
            volume_today = int(hist.select("volume").to_series()[0])
            avg_volume_20d = sum(map(int, hist.select("volume").to_series().to_list())) / len(
                recent_closes
            )

            if (
                abs((today_high - high_20d) / high_20d) > breakout_threshold
                or abs((low_20d - today_low) / low_20d) > breakout_threshold
            ) and volume_today >= avg_volume_20d * volume_multiplier:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n_breakers = sorted(breakout_symbols)[: self._top_n]
        weight = 1.0 / len(top_n_breakers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_breakers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest