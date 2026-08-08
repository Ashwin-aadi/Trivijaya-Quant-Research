from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityBasedStrategy(Strategy):
    rationale = (
        "The strategy exploits historical patterns where certain stocks exhibit higher returns "
        "during specific months or days of the year. It focuses on periods like October and November "
        "due to Diwali effects and December due to year-end buying, combined with specific weekdays."
    )

    def __init__(self, window: int = 3, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Filter by month and day of the week
        filtered_history = history.select(
            pl.col("session_date").dt.month_name(),
            pl.col("session_date").dt.weekday(),
            *symbols,
        ).filter(
            (pl.col("session_date").dt.month_name().is_in(["October", "November", "December"])) &
            (pl.col("session_date").dt.weekday() < 5)  # Monday to Friday
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        monthly_closes = filtered_history.groupby("session_date").agg(
            *[pl.col(symbol).mean().alias(symbol) for symbol in symbols]
        ).sort("session_date")

        # Calculate RSI
        monthly_returns = monthly_closes.select([f"close_{i}" for i in range(1, self._window + 1)])
        relative_strengths = (monthly_returns.shift(-1) / monthly_returns - 1.0).to_dict()

        rsi_scores = {
            symbol: max((sum(relative_strengths[symbol][:-1] > 0) - sum(relative_strengths[symbol][:-1] < 0)) / self._window, 0)
            for symbol in symbols
        }

        # Bollinger Bands
        upper_bands = (monthly_closes.select(symbols) + 2 * monthly_closes.select([f"close_{i}" for i in range(1, 3)]).std()).to_dict()
        lower_bands = (monthly_closes.select(symbols) - 2 * monthly_closes.select([f"close_{i}" for i in range(1, 3)]).std()).to_dict()

        # Select top N symbols based on RSI and Bollinger Bands
        final_scores = {
            symbol: rsi_scores[symbol] + (upper_bands[symbol][-1] > monthly_closes["close"].max())
            for symbol in symbols
        }
        sorted_symbols = sorted(final_scores.items(), key=lambda x: -x[1])[:self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest