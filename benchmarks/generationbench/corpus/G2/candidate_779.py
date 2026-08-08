from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining a momentum indicator with a volatility measure aims to capture both "
        "trend-following and risk-adjusted signals. Strong trends often precede increased "
        "volatility, which can be an early warning sign of trend reversals."
    )

    def __init__(self, momentum_window: int = 10, volatility_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = []
        for symbol in view.symbols:
            close_series = history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol))["adj_close"]
            if close_series.height < self._momentum_window:
                continue
            mom_score = (close_series[-1] - close_series[self._momentum_window - 1]) / abs(close_series[self._momentum_window - 1])
            momentum_scores.append((symbol, mom_score))

        volatility_scores = []
        for symbol in view.symbols:
            vol_series = history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol))["adj_close"]
            if vol_series.height < self._volatility_window:
                continue
            daily_returns = (vol_series / vol_series.shift(1) - 1).drop_nulls()
            volatility_score = daily_returns.std().item()
            volatility_scores.append((symbol, volatility_score))

        momentum_ranks = {s: v for s, v in sorted(momentum_scores, key=lambda x: abs(x[1]), reverse=True)}
        volatility_ranks = {s: v for s, v in sorted(volatility_scores, key=lambda x: x[1])}

        combined_scores = {}
        for symbol in momentum_ranks.keys():
            if symbol not in volatility_ranks:
                continue
            combined_score = (momentum_ranks[symbol] + 2 * volatility_ranks[symbol]) / 3
            combined_scores[symbol] = combined_score

        top_symbols = sorted(combined_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        weights = {s: 0.2 for s, _ in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest