from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of their initial move. By identifying "
        "breakouts that continue to trend and assigning weights accordingly, we aim to capture "
        "the residual momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).select(
            "symbol", pl.col("session_date").alias("date"), "adj_close"
        )
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(closes) < self._window + 1:
                continue

            # Check if the last close is a breakout and continues in that direction
            last_breakout_close = max(closes[:-1]) if closes[-1] >= max(closes[:-1]) else min(closes[:-1])
            if (
                (closes[-1] > last_breakout_close and all(c < last_breakout_close for c in closes[1:-1])) or
                (closes[-1] < last_breakout_close and all(c > last_breakout_close for c in closes[1:-1]))
            ):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected symbol
        weight = 1.0 / len(breakout_symbols)
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