from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit seasonal behavior "
        "due to factors like earnings releases or holidays. This strategy aims to "
        "capitalize on these patterns by identifying and investing in stocks that "
        "tend to perform well during specific times of the year."
    )

    def __init__(self, window: int = 365, lookback: int = 2) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or (history.height - 1 < self._lookback):
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            closes = [float(v) for v in history.select(pl.col(symbol)).to_series().drop_nulls().to_list()]
            monthly_returns = [
                (closes[i + 1] - closes[i]) / closes[i]
                for i in range(len(closes) - 1)
            ]
            recent_returns = sorted(monthly_returns[-self._lookback:], reverse=True)[:2]

            seasonality_factors[symbol] = sum(recent_returns)

        top_symbols = sorted(seasonality_factors, key=seasonality_factors.get, reverse=True)[:3]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest