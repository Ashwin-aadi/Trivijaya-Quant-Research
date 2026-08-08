from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After an initial breakout, a continuation of the breakout direction is often "
        "indicative of a sustained move. This strategy looks for symbols that have already"
        " broken out and are continuing in the same direction."
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
            if symbol not in history["symbol"].to_list():
                continue
            last_close = float(history.filter(pl.col("session_date") == stamp).get(symbol))
            history_subset = history.filter(pl.col("symbol") == symbol)
            breakout_price = (
                history_subset.select(pl.col("adj_close").max())
                .sort(by="session_date", descending=True)
                .head(1)
                .get(0)
            )
            if not breakout_price:
                continue
            breakout_price = float(breakout_price)
            last_close_to_breakout_ratio = last_close / breakout_price - 1.0

            if last_close_to_breakout_ratio >= 0.05:  # Consider a small buffer for continuation
                previous_direction = (
                    history_subset.sort(by="session_date")
                    .select(
                        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
                    )
                    .head(self._lookback)
                    .get_column("r")
                    .to_list()
                )
                if all(r > 0 for r in previous_direction):
                    breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

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