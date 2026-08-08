from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy exploits seasonality in the Indian equity market by identifying "
        "historical patterns that indicate higher returns during certain months and industries."
    )

    def __init__(self, long_term_lookback: int = 10, short_term_lookback: int = 3) -> None:
        self._long_term_lookback = long_term_lookback
        self._short_term_lookback = short_term_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_term_lookback)
        if closes.height < self._long_term_lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate average returns for each month
        monthly_returns: dict[str, float] = {}
        current_month = stamp.month

        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._long_term_lookback:
                continue
            monthly_values = [
                (values[i + 1] - values[i]) / values[i]
                for i in range(len(values) - 1)
            ]
            avg_return = sum(monthly_values[-self._short_term_lookback:]) / len(
                monthly_values[-self._short_term_lookback:]
            )
            if current_month == stamp.month:
                monthly_returns[symbol] = avg_return

        # Rank symbols based on average returns
        ranked_symbols = sorted(
            monthly_returns.items(), key=lambda x: x[1], reverse=True
        )

        top_n_symbols = [symbol for symbol, _ in ranked_symbols[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest