from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the high-low range of a stock's price narrows over "
        "a period. This can indicate that extreme prices are being bid away or sold at a more "
        "efficient rate, suggesting an upcoming breakout in either direction. By identifying "
        "such stocks, we aim to benefit from the subsequent price action."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].unique()) < 2:
                continue
            high_low_range = (
                (history[f"{symbol}_high"] - history[f"{symbol}_low"])
                .sort(descending=True)
                .head(1)[0]
            )
            recent_high_low_ratio = (
                history[f"{symbol}_close"].max() / history[f"{symbol}_close"].min()
            )

            if high_low_range < 0.5 * (history[f"{symbol}_high"] - history[f"{symbol}_low"]).mean():
                picks.append(symbol)

        picks = picks[:10]
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