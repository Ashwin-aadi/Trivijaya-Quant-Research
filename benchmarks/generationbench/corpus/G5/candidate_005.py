from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. This strategy identifies "
        "breakout candidates and assigns weights to those that have a strong momentum post-breakout."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 2 + 10)  # Ensure enough data for robust analysis
        if history.height < (self._window * 3):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_close_ratio = (
                history.filter(
                    (pl.col("session_date") >= pl.col("session_date").first())
                    & (pl.col("session_date") < pl.col("session_date").last()).shift(1)
                )["close"]
                / history.filter(pl.col("session_date") == pl.col("session_date").last().shift(1))[
                    "adj_close"
                ]
            ).to_list()
            if len(high_close_ratio) > self._window:
                ratio = max(high_close_ratio[-self._window:])
                if ratio >= 1.05:  # Consider a breakout
                    breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        continuation = view.history().filter(pl.col("symbol").is_in(breakout_symbols))
        continuation_values = (
            continuation.filter(
                (pl.col("session_date") > pl.col("session_date").last().shift(1))
                & (pl.col("session_date") <= view.as_of)
            )["close"]
            / continuation.filter(pl.col("session_date") == pl.col("session_date").last().shift(2))[
                "adj_close"
            ]
        ).to_list()

        momentum_scores = [v - 1 if v >= 1 else 0 for v in continuation_values]

        top_symbols = sorted(
            breakout_symbols,
            key=lambda x: momentum_scores[breakout_symbols.index(x)],
            reverse=True,
        )[: self._top_n]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest