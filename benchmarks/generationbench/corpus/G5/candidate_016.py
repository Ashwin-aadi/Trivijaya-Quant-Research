from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels revert to the mean over time. By identifying symbols that have deviated "
        "significantly from their trailing average, we can generate trading signals based on "
        "reverting back towards this mean. This strategy aims to exploit mean reversion by "
        "focusing on strong deviations."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_prices: dict[str, float] = {
            symbol: float(closes[symbol].mean().item())
            for symbol in view.symbols
        }

        std_prices: dict[str, float] = {
            symbol: float(closes[symbol].std().item())
            for symbol in view.symbols
        }

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_prices or symbol not in std_prices:
                continue

            latest_close = float(view.latest_close()[symbol])
            z_score = (latest_close - mean_prices[symbol]) / std_prices[symbol]

            if abs(z_score) > self._z_score_threshold:  # Consider only strong deviations
                weights[symbol] = 1.0 / len(weights)

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest