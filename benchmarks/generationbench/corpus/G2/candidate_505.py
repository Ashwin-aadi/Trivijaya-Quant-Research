from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when financial assets return to their historical mean price. "
        "In the short term, stocks that have been unusually high or low relative to their history "
        "tend to revert towards the mean. This strategy aims to profit from such deviations."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            daily_returns = (
                (history.select(pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0))
                .collect()
                .with_column(
                    pl.when(pl.col("session_date") == stamp)
                    .then(pl.col("r"))
                    .otherwise(None)
                    .alias("score")
                )
                .filter(pl.col("score").is_not_null())
            )
            if daily_returns.height < 1:
                continue
            score = (
                (daily_returns.select(pl.col("score")).mean().to_list()[0] * -1.0) + 1.0
            )  # Normalize to [0, 1]
            mean_reversion_scores[symbol] = score

        if not mean_reversion_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(mean_reversion_scores.keys(), key=lambda x: mean_reversion_scores[x], reverse=True)[:5]
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