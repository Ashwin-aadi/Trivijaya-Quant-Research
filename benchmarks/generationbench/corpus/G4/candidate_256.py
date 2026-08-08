from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion200d(Strategy):
    rationale = (
        "Price-level reversion suggests that stock prices tend to return to their historical "
        "averages over time. By identifying stocks with significant deviations from these averages, "
        "we can exploit the tendency for prices to revert towards this mean."
    )

    def __init__(self, window: int = 200, top_n_undervalued: int = 15, top_n_overvalued: int = 15) -> None:
        self._window = window
        self._top_n_undervalued = top_n_undervalued
        self._top_n_overvalued = top_n_overvalued

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Compute the trailing moving average
        sma_col = f"sma_{self._window}"
        history = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window=self._window)).alias(sma_col)
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Calculate price deviation
        history = history.with_columns(
            (pl.col("adj_close") / pl.col(sma_col) - 1).alias("deviation")
        )

        # Rank candidates based on deviation
        ranks = (
            history.select(pl.exclude(["symbol", "session_date"])).to_numpy().tolist()
        )
        ranked_indices = [i for _, i in sorted(zip(ranks, range(len(ranks))), reverse=True)]

        undervalued_symbols = [
            view.symbols[i] for i in ranked_indices[: self._top_n_undervalued]
        ]
        overvalued_symbols = [
            view.symbols[i] for i in ranked_indices[-self._top_n_overvalued :]
        ]

        weights = {}
        if undervalued_symbols:
            weight = 1.0 / len(undervalued_symbols)
            for symbol in undervalued_symbols:
                weights[symbol] = weight
        elif overvalued_symbols:
            weight = -1.0 / len(overvalued_symbols)
            for symbol in overvalued_symbols:
                weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights={s: float(w) for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_pydatetime().date()
    assert isinstance(newest, date)
    return newest