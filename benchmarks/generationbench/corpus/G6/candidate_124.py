from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion200d(Strategy):
    rationale = (
        "Leverage mean reversion by comparing current stock prices with their historical "
        "average over 200 days. Stocks that deviate significantly from this historical average"
        " are considered mispriced and provide entry opportunities."
    )

    def __init__(self, window: int = 200, threshold: float = 3.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = [col for col in closes.columns if col not in ["session_date"]]
        avg_close = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("avg_close"))
                   .to_pandas()
        )

        def reversion_score(closes: pl.DataFrame, symbol: str) -> float:
            latest_close = closes.height - 1
            mean_close = avg_close.loc[avg_close["symbol"] == symbol, "avg_close"].iloc[0]
            std_dev = (
                history.filter(pl.col("symbol") == symbol)
                        .group_by("symbol")
                        .agg(pl.col("adj_close").std().alias("std_dev"))
                        .to_pandas()
                        .loc[0, "std_dev"]
            )
            return (closes["close"][latest_close] - mean_close) / std_dev

        scores = [
            reversion_score(closes, symbol)
            for symbol in symbols
            if symbol in closes.columns and not closes[symbol].to_list()[-1].is_nan()
        ]

        extreme_scores: list[str] = []
        for symbol in symbols:
            score = reversion_score(closes, symbol)
            if abs(score) > self._threshold:
                extreme_scores.append(symbol)

        weight = 1.0 / len(extreme_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in extreme_scores}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest