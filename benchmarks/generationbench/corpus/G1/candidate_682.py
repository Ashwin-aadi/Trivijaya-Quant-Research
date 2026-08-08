from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian market can be driven by various factors such as "
        "holiday effects or fiscal year end influences. We exploit this by identifying "
        "symbols that have historically performed well during certain times of the year."
    )

    def __init__(self, window: int = 365, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            close_prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].drop_nulls().to_list()]
            if len(close_prices) < self._window:
                continue

            returns = [
                (close_prices[i] / close_prices[i - 1] - 1.0)
                for i in range(1, len(close_prices))
            ]
            seasonal_returns[symbol] = max(returns)

        sorted_returns = sorted(seasonal_returns.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_returns[:self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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