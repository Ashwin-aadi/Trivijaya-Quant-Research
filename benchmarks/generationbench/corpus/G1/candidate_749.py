from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies look for stocks that have already broken out of their "
        "recent range and then continue to move in the direction of the breakout. This can signal "
        "a sustained trend and potentially higher returns."
    )

    def __init__(self, window: int = 20, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_continuation_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            recent_closes = history.filter(pl.col("symbol") == symbol).select(
                "close"
            ).sort("session_date").to_pandas()["close"].tolist()
            if len(recent_closes) < self._window:
                continue

            breakout_close = max(recent_closes)
            breakout_threshold = breakout_close * (1 + self._threshold)

            for i in range(self._window, 0, -1):
                if recent_closes[-i] >= breakout_threshold and (
                    len(breakout_continuation_symbols) < 5
                ):
                    breakout_continuation_symbols.append(symbol)
                    break

        breakout_continuation_symbols = list(set(breakout_continuation_symbols))
        if not breakout_continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest