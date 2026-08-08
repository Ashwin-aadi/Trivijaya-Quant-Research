from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTradeStrategy(Strategy):
    rationale = (
        "Seasonality in stock markets can arise due to predictable patterns in corporate earnings, "
        "government policies, or cultural events. For example, certain sectors might perform well "
        "during specific times of the year based on historical trends."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Compute daily returns
            daily_returns = [(values[i] - values[i-1]) / values[i-1] if i > 0 else 0.0 for i in range(len(values))]

            # Filter out dates where we don't have enough data to make a decision
            valid_returns = [r for i, r in enumerate(daily_returns) if i >= self._window - 1 and i < len(daily_returns)]

            if not valid_returns:
                continue

            mean_return = sum(valid_returns) / len(valid_returns)

            # Identify symbols with above average returns
            if values[-1] > max(values):
                picks.append(symbol)
        
        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest