from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in India may exhibit seasonal trends where their performance "
        "is influenced by calendar events such as festivals or holidays. By identifying "
        "these seasonal patterns, we can potentially capture higher returns during favorable "
        "periods."
    )

    def __init__(self, festival_window: int = 30) -> None:
        self._festival_window = festival_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=5 * self._festival_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_prices) < self._festival_window * 5:
                continue

            # Calculate the average return around the festival period
            festival_dates = [
                date.fromordinal(int(d)) for d in pl.Series("session_date", history["session_date"]).to_numpy()
            ]
            festival_period_returns: list[float] = []
            for i, date_ in enumerate(festival_dates):
                start = max(0, i - 2)
                end = min(len(close_prices), i + 3)
                if end - start > 1:
                    return_ = (close_prices[end - 1] / close_prices[start]) - 1
                    festival_period_returns.append(return_)
            
            if len(festival_period_returns) < 5:
                continue

            avg_return = sum(festival_period_returns) / len(festival_period_returns)
            seasonal_factors[symbol] = avg_return

        # Select top performing symbols based on the average returns during the festival period
        sorted_symbols = [k for k, v in sorted(seasonal_factors.items(), key=lambda item: -item[1])]
        top_symbols = sorted_symbols[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest