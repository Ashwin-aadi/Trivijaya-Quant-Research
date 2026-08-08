from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves identifying trending symbols and "
        "allocating capital to them. Symbols with higher recent volatility are more likely "
        "to continue their trend direction."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        volatility_scores: dict[str, float] = {}
        for symbol in symbols:
            daily_returns = (
                (history.select(pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1.0)
                .filter((pl.col("symbol") == symbol))
                .to_series()
                .to_list()
            )
            if len(daily_returns) < self._window:
                continue
            volatility = (sum(abs(r) for r in daily_returns) / self._window) ** self._scaling_factor
            volatility_scores[symbol] = volatility

        top_symbols = sorted(volatility_scores, key=volatility_scores.get, reverse=True)[:5]
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