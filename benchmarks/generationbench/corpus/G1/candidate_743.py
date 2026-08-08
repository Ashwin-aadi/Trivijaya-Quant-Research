from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day momentum "
        "and the 5-day volatility. By combining these, we aim to identify stocks that are "
        "both strong in momentum and stable in recent days."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        volatility_scores: list[float] = []

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(adj_closes) < self._momentum_window + self._volatility_window:
                continue

            momentum_score = (adj_closes[-1] - adj_closes[0]) / max(adj_closes)
            volatility_score = pl.col("adj_close").std().over(pl.arange(1, self._volatility_window + 1)).mean().to_list()[-1]

            momentum_scores.append(momentum_score)
            volatility_scores.append(volatility_score)

        top_momentum_symbols = [symbol for _, symbol in sorted(zip(momentum_scores, view.symbols), reverse=True)[:5]]
        top_volatility_symbols = [symbol for _, symbol in sorted(zip(volatility_scores, view.symbols), reverse=False)[:3]]

        common_symbols = set(top_momentum_symbols).intersection(set(top_volatility_symbols))

        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(common_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in common_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest