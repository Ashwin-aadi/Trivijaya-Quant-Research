from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, a consolidation period often follows. If the market does "
        "not break through the previous high or low from before the breakout, it may indicate "
        "a continuation of the trend rather than a reversal. This strategy identifies such "
        "continuations and bets on the ongoing trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_highs = []
        for symbol in view.symbols:
            high_series = history.select(pl.col("symbol").filter(pl.col("symbol") == symbol).select("high")).to_dict(
                "records"
            )
            highs = [float(v["high"]) for v in high_series]
            if len(highs) < self._window + 1:
                continue
            breakout_high = max(highs[:-1])
            current_high = highs[-1]
            if current_high > breakout_high * 1.05:  # Slight buffer to avoid false positives
                breakout_highs.append(symbol)

        if not breakout_highs:
            return Signal(information_available_at=stamp, weights={})

        continuation_symbols = []
        for symbol in breakout_highs:
            history_df = view.history(lookback=self._window)
            last_close = float(history_df.filter(pl.col("symbol") == symbol).select("adj_close").to_dict("records")[0][
                "adj_close"
            ])
            if (
                (last_close >= max(history_df.filter(pl.col("symbol") == symbol).select("high")).to_dict(
                    "records"
                )[0]["high"]
                 )
                or (last_close <= min(history_df.filter(pl.col("symbol") == symbol).select("low")).to_dict(
                    "records"
                )[0]["low"])
            ):
                continuation_symbols.append(symbol)

        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(continuation_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in continuation_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest