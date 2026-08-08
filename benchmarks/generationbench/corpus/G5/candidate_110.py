from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on volatility. High recent volatility suggests "
        "that the market is uncertain and potentially due for a trend. We enter long positions "
        "on symbols with high positive returns to anticipate an upward trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        volatilities: dict[str, float] = {}

        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            closes = df.select("adj_close").to_list()
            returns = [(closes[i + 1] - closes[i]) / closes[i] if i < len(closes) - 1 else 0.0 for i in range(len(closes))]
            positive_returns = sum(r > 0 for r in returns)
            volatility = (sum(abs(r) for r in returns) / self._window) ** 0.5
            mean_return = sum(r for r in returns if r > 0) / max(positive_returns, 1)
            volatilities[symbol] = float(volatility * mean_return)

        sorted_volatilities = {k: v for k, v in sorted(volatilities.items(), key=lambda item: item[1], reverse=True)}
        top_symbols = list(sorted_volatilities.keys())[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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