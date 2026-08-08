from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrend(Strategy):
    rationale = (
        "Trend following strategies exploit periods of high market volatility by "
        "entering trends before they reverse. High volatility often precedes trend "
        "reversals, so entering positions in highly volatile stocks can lead to "
        "positive returns."
    )

    def __init__(self, window: int = 20, threshold_volatility: float = 1.5) -> None:
        self._window = window
        self._threshold_volatility = threshold_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_scores = {}
        for symbol in view.symbols:
            daily_returns = (
                history.select([pl.col("adj_close"), pl.col("session_date")])
                .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
                .sort("session_date")
                .filter(pl.col("symbol") == symbol)
            )
            if daily_returns.height < self._window:
                continue
            mean_return = daily_returns.select(pl.col("r").mean().alias("m")).row(0)[0]
            std_deviation = daily_returns.select(pl.col("r").stddev().alias("sd")).row(0)[0]
            volatility_score = (std_deviation / abs(mean_return)) if mean_return != 0 else 1.0
            volatility_scores[symbol] = volatility_score

        high_volatility_symbols = [
            symbol for symbol, score in volatility_scores.items() if score > self._threshold_volatility
        ]
        if not high_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volatility_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volatility_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest