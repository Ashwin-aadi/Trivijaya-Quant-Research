from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following strategies aim to capture trends by holding assets that have "
        "recently shown increasing volatility. This can help in identifying strong trends and "
        "potentially profitable opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(adj_closes) < self._window + 1:
                continue

            prices = pl.Series(adj_closes)
            log_returns = (prices / prices.shift(1).fill_null(1.0) - 1.0).drop_nulls()
            volatility_score = log_returns.std().to_numpy()[0]
            volatility_scores[symbol] = volatility_score

        symbols_to_trade: list[str] = sorted(volatility_scores.keys(), key=lambda k: volatility_scores[k], reverse=True)
        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest