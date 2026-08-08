from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Stocks often exhibit seasonal patterns driven by economic activities or investor "
        "behavior. This strategy identifies and exploits these trends by buying stocks that "
        "have historically performed well during specific months."
    )

    def __init__(self, window: int = 20, lookback_months: int = 12) -> None:
        self._window = window
        self._lookback_months = lookback_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_months)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonality_trends = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_history = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_close_history) < self._lookback_months * 20:
                continue

            monthly_returns = [
                (adj_close_history[i + 20] - adj_close_history[i]) / adj_close_history[i]
                for i in range(0, len(adj_close_history), 20)
            ]
            avg_return = sum(monthly_returns) / max(len(monthly_returns), 1)
            seasonality_trends[symbol] = avg_return

        top_symbols = sorted(seasonality_trends.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [t[0] for t in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest