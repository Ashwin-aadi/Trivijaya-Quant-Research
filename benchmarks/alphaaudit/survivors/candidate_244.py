from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion is a principle that after an extreme price move in one direction, "
        "the price will tend to revert back towards its mean. This strategy identifies symbols"
        " that have moved significantly and bets on their reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_price = sum(values) / self._window
            std_deviation = (sum((v - mean_price) ** 2 for v in values) / self._window) ** 0.5

            recent_close = values[-1]
            z_score = (recent_close - mean_price) / std_deviation if std_deviation > 0 else 0
            reversion_scores[symbol] = abs(z_score)

        top_symbols = sorted(reversion_scores, key=reversion_scores.get, reverse=True)[:5]
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