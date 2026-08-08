from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining a short-term momentum signal with a medium-term moving average crossover "
        "could provide a balanced approach that leverages both trend and mean reversion. "
        "Short-term momentum can capture fast market movements, while the moving average "
        "crossover offers insights into longer-term trends."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._short_window + self._long_window)

        if closes.height < self._short_window + self._long_window:
            return Signal(information_available_at=stamp, weights={})

        short_mavg = (closes["close"] / closes["close"].rolling_mean(window_size=self._short_window).shift(1) - 1.0).alias("r")
        long_mavg = (closes["close"] / closes["close"].rolling_mean(window_size=self._long_window).shift(1) - 1.0).alias("r")

        mavg_df = (
            closes
                .with_columns(short_mavg, long_mavg)
                .filter(pl.col("session_date") == stamp)
                .select([
                    "symbol",
                    (pl.col("short_mavg") - pl.col("long_mavg")).alias("composite_signal")
                ])
        )

        if mavg_df.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            mavg_df.sort("composite_signal", descending=True)
                .select(["symbol"])
                .head(5)["symbol"]
                .to_list()
        )

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest