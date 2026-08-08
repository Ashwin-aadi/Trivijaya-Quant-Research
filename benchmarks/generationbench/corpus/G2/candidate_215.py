from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines a momentum-based entry with a volatility filter. "
        "Momentum suggests that stocks which have outperformed recently are more likely to continue outperforming, "
        "while low volatility can indicate reduced downside risk and higher return potential."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._volatility_window)

        if closes.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        volatility_scores: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]

            # Calculate momentum score as the percentage change from the mean of last 20 days to today.
            if len(close_values) < self._momentum_window + 1:
                continue
            recent_mean = sum(close_values[-self._momentum_window:]) / self._momentum_window
            current_close = close_values[-1]
            momentum_score = (current_close - recent_mean) / recent_mean
            momentum_scores.append(momentum_score)

            # Calculate volatility score as the standard deviation of the last 10 days.
            if len(close_values) < self._volatility_window + 1:
                continue
            vol_score = pl.DataFrame({"close": close_values[-self._volatility_window:]}).select(
                (pl.col("close").std()).alias("volatility")
            ).item()
            volatility_scores.append(vol_score)

        momentum_rank = sorted(range(len(momentum_scores)), key=lambda k: momentum_scores[k], reverse=True)
        volatility_rank = sorted(range(len(volatility_scores)), key=lambda k: volatility_scores[k])

        picks: list[str] = []
        for symbol_index in momentum_rank[:5]:
            if volatility_scores[symbol_index] < 0.1 and close_values[-1] > max(close_values[-self._momentum_window:-1]):
                picks.append(view.symbols[symbol_index])

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest