from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying symbols where the current price "
        "is far from their trailing average, we can generate buy or sell signals based on "
        "this tendency."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        trailing_means: pl.DataFrame = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("trailing_mean")))
            .select(["symbol", "trailing_mean"])
        )

        closes: pl.DataFrame = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        reversion_scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in trailing_means.column_names or symbol not in closes.column_names:
                continue
            adj_close_series = [float(v) for v in closes[symbol].to_list()]
            trailing_mean = float(trailing_means.filter(pl.col("symbol") == symbol).select("trailing_mean").item())
            score = abs(adj_close_series[-1] - trailing_mean)
            reversion_scores.append((symbol, score))

        reversion_scores.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in reversion_scores[:5]]
        weight = 1.0 / len(top_symbols) if top_symbols else 0.0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest