from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the price action is consolidating in a narrow range, "
        "which often precedes a breakout. By identifying symbols with significant recent range "
        "compression, we can capture potential breakouts."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            daily_data = history.filter(pl.col("symbol") == symbol)
            price_range = (daily_data.select("high").max() - daily_data.select("low").min()).to_list()[0]
            if price_range <= 0:
                continue
            recent_high = daily_data.sort("session_date", descending=True).head(2)["high"].max()
            recent_low = daily_data.sort("session_date", descending=True).head(2)["low"].min()
            recent_range = recent_high - recent_low

            if recent_range / price_range < 0.5:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = sorted(signals.keys(), key=lambda k: -signals[k])[:self._top_n]
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest