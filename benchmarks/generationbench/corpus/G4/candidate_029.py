from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken above a key resistance level "
        "and aims to capitalize on the breakout continuation pattern. By setting buy orders just "
        "above the breakout level with stop-loss and take-profit levels, we aim to capture upward "
        "momentum while managing risk."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_scores = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            # Calculate HH and LL over the last window days
            hh = float(history.filter(pl.col("session_date") < pl.lit(view.as_of)).group_by("symbol").agg(
                (pl.col("high").max().alias("hh"))
            ).with_columns((pl.col("hh") == history.filter(pl.col("symbol") == pl.col("symbol")).select("adj_close"))).filter(
                pl.col("hh") == True
            )["hh"].to_list()[0])

            ll = float(history.filter(pl.col("session_date") < pl.lit(view.as_of)).group_by("symbol").agg(
                (pl.col("low").min().alias("ll"))
            ).with_columns((pl.col("ll") == history.filter(pl.col("symbol") == pl.col("symbol")).select("adj_close"))).filter(
                pl.col("ll") == True
            )["ll"].to_list()[0])

            # Check if the latest close is above HH
            if float(history.select("adj_close").filter(pl.col("symbol") == symbol)[-1]) > hh:
                score = 1.0
            else:
                score = 0.0

            breakout_scores.append((symbol, score))

        # Rank stocks based on breakout scores
        ranked_symbols = sorted(breakout_scores, key=lambda x: x[1], reverse=True)[:self._top_n]
        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [sym[0] for sym in ranked_symbols]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest