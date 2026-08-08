from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuance of a breakout can be interpreted as persistent positive sentiment. "
        "If a stock breaks out to new highs and continues to rise for several sessions, it may "
        "indicate strong buying pressure and could provide an entry point."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.is_empty() or history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            session_dates = [d for d in history["session_date"].to_list()]
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]

            breakout_date = max(range(len(adj_closes)), key=lambda i: adj_closes[i])
            if (
                breakout_date >= len(adj_closes) - self._lookback
                or adj_closes[breakout_date] <= min(adj_closes[-self._lookback:])
            ):
                continue

            for date_index in range(breakout_date, len(adj_closes)):
                current_close = adj_closes[date_index]
                previous_close = adj_closes[date_index - 1]
                if (
                    current_close > previous_close
                    and session_dates[date_index] < view.as_of
                ):
                    breakout_symbols.append(symbol)
                    break

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest