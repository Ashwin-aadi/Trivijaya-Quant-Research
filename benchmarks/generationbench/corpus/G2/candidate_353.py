from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often lead to continuation patterns. If a stock breaks out of its recent range "
        "and continues in the breakout direction over the next few sessions, it is likely to offer "
        "profit opportunities. This strategy identifies such stocks and invests in them."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_series().to_list():
                continue

            # Get the last 'n' days of closes
            last_n_days = history.filter(
                pl.col("session_date") >= (view.as_of - date(self._window, 1, 1))
            )

            # Compute daily returns
            last_n_days = last_n_days.with_columns(
                (
                    pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0
                ).alias("return")
            )
            last_n_days = last_n_days.sort("session_date", descending=False)

            if last_n_days.height < self._window:
                continue

            # Find the breakout day
            for i in range(self._window, last_n_days.height):
                if last_n_days["return"][i] > 0.01:  # Assuming a significant return threshold
                    break
            else:
                continue

            # Check continuation pattern over next 'm' days
            continuation_days = history.filter(
                (pl.col("session_date") >= (last_n_days.select("session_date")[i] + date(1, 0, 0)))
                & (pl.col("session_date") < (view.as_of - date(self._continuation_window, 1, 1)))
            )
            if continuation_days.height < self._continuation_window:
                continue

            # Ensure the stock continues in breakout direction
            if all(
                last_n_days["return"][i + j] > 0.005 for j in range(self._continuation_window)
            ):
                breakout_symbols.append(symbol)

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