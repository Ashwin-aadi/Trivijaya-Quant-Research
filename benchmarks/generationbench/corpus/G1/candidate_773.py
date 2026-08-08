from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation of a breakout suggests that the price move is likely to continue in "
        "the same direction. By identifying stocks that have broken out and continued moving "
        "in that direction, we can capitalize on this trend."
    )

    def __init__(self, window: int = 20, lookback_for_breakout: int = 5) -> None:
        self._window = window
        self._lookback_for_breakout = lookback_for_breakout

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback_for_breakout)

        if history.height < self._window + self._lookback_for_breakout:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue

            window_close = (
                history.filter(pl.col("symbol") == symbol)
                         .sort(by="session_date")
                         .select(pl.col("close").tail(self._window))
                         .collect()
                         .get_column(0)
                         .to_list()
            )

            breakout_window = history.filter(
                (pl.col("symbol") == symbol) & (
                    pl.col("session_date") >= date.today() - self._lookback_for_breakout
                )
            ).select(pl.col("close"))

            if breakout_window.height < 1:
                continue

            last_close = float(breakout_window.sort(by="session_date").select(pl.last()).item())
            if window_close[-1] > max(window_close) and last_close > max(window_close):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest