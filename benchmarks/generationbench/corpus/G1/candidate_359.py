from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy identifies stocks with both strong short-term and long-term momentum. "
        "Such stocks are likely to continue trending positively in the near future."
    )

    def __init__(self, short_window: int = 10, long_window: int = 50) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window)

        if history.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_returns = (history["adj_close"] / history["adj_close"].shift(self._short_window) - 1.0).alias("short_ret")
        long_returns = (history["adj_close"] / history["adj_close"].shift(self._long_window) - 1.0).alias("long_ret")

        momentum_df = history.with_columns(short_returns, long_returns)
        momentum_df = momentum_df.sort("session_date", descending=True).head(20)

        top_short_moments: list[str] = []
        for symbol in view.symbols:
            if symbol not in momentum_df.columns or "short_ret" not in momentum_df[symbol].to_list():
                continue
            short_values = [float(v) for v in momentum_df[f"{symbol}.short_ret"].drop_nulls().to_list()]
            top_short_moments.append(symbol)

        top_long_moments: list[str] = []
        for symbol in view.symbols:
            if symbol not in momentum_df.columns or "long_ret" not in momentum_df[symbol].to_list():
                continue
            long_values = [float(v) for v in momentum_df[f"{symbol}.long_ret"].drop_nulls().to_list()]
            top_long_moments.append(symbol)

        combined_picks: list[str] = set(top_short_moments).intersection(set(top_long_moments))
        if not combined_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in combined_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest