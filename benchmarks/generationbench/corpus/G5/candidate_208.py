from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reverts against a trailing reference level. "
        "This approach captures mean-reverting behavior in the stock price."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("m"))
            .select(["symbol", "m"])
            .with_columns((pl.col("adj_close") - pl.col("m")).alias("deviation"))
            .sort("deviation", descending=True)
            .head(self._top_n())
        )

        top_symbols = mean_close.select("symbol").to_series().to_list()
        weights = self._compute_weights(top_symbols, history)

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in zip(top_symbols, weights)}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


class ReversionTrailing(Strategy):
    def _top_n(self) -> int:
        # Adjust the number of top symbols based on the window size
        return min(self._window // 2, len(view.symbols))


def _compute_weights(self, symbols: list[str], history: pl.DataFrame) -> list[float]:
    weights = []
    for symbol in symbols:
        latest_close = float(history.filter(pl.col("symbol") == symbol).select("adj_close").head(1)["adj_close"])
        mean_close = float(history.group_by("symbol").agg(pl.col("adj_close").mean()).filter(pl.col("symbol") == symbol).select("m")[0, 1])
        deviation = latest_close - mean_close
        weight = max(0.0, min(self._threshold * (self._window / len(symbols)), abs(deviation)))
        weights.append(weight)
    return weights