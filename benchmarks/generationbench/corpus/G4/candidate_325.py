from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when stock prices tend to revert to their historical average levels. "
        "This strategy identifies stocks that have deviated significantly from their moving averages and "
        "capitalizes on the tendency for these deviations to correct."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = history[symbols].select(
            pl.col("adj_close").alias(f"close_{symbol}")
            for symbol in symbols
        )
        sma = (
            closes.sort("session_date")
            .group_by("symbol", maintain_order=True)
            .agg((pl.col(f"close_{{}}".format(s)) / pl.col(f"close_{{}}".format(s)).shift(1) - 1.0).alias(f"r"))
            .with_columns(pl.col(f"r").mean().over("symbol", by="session_date").alias(f"sma_{self._window}d"))
        )
        closes = (
            sma.join(history, on=["symbol"], how="inner")
            .select(
                [pl.col("adj_close"), f"sma_{self._window}d"]
                + [pl.col(f"close_{{}}".format(s)) for s in symbols]
            )
            .sort("session_date", descending=False)
        )

        scores = [
            (symbol, float(closes[f"close_{symbol}"][-1] - closes[f"sma_{self._window}d"][-1]))
            for symbol in symbols
        ]
        ranked_scores = sorted(scores, key=lambda x: abs(x[1]), reverse=True)

        picks: list[str] = [ranked_scores[i][0] for i in range(min(self._threshold * 2, len(ranked_scores)))]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight
                for s in picks
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest