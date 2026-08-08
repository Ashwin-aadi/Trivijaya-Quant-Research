from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum signals can capture both the "
        "immediate trend strength and the sustained price movement of stocks."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._short_window + self._long_window - 1)
        if closes.height < self._short_window + self._long_window - 1:
            return Signal(information_available_at=stamp, weights={})

        short_momentum = (closes[closes.columns[0]] / closes[closes.columns[0]].shift(self._short_window) - 1.0).alias("short_r")
        long_momentum = (closes[closes.columns[0]] / closes[closes.columns[0]].shift(self._long_window) - 1.0).alias("long_r")

        short_df = closes.with_columns(short_momentum)
        long_df = closes.with_columns(long_momentum)

        short_top_gainers = [symbol for symbol in view.symbols if float((short_df[short_df["session_date"] == stamp].select(short_momentum)[symbol]) > 0.05)]
        long_top_gainers = [symbol for symbol in view.symbols if float((long_df[long_df["session_date"] == stamp].select(long_momentum)[symbol]) > 0.1)]

        common_symbols = set(short_top_gainers) & set(long_top_gainers)

        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(common_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in common_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest