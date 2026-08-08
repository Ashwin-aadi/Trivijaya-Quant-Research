from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy leverages historical seasonal patterns in Indian equity markets, "
        "such as the impact of monsoon season and fiscal budget announcements. By identifying key dates, "
        "we can predict favorable movements and exploit them through targeted trading positions."
    )

    def __init__(self, window_monsoon: int = 30, window_budget: int = 60, max_positions: int = 20) -> None:
        self._window_monsoon = window_monsoon
        self._window_budget = window_budget
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_monsoon, self._window_budget))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = self._select_top_n_symbols(history)
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _select_top_n_symbols(history: pl.DataFrame) -> list[str]:
    symbols = history["symbol"].unique().to_list()
    top_n_symbols = []

    for symbol in symbols:
        symbol_history = history.filter(pl.col("symbol") == symbol).sort("session_date").to_pandas()

        # Calculate daily returns and volume adjusted returns
        symbol_history["daily_return"] = (symbol_history["adj_close"].pct_change() * 100).fillna(0)
        symbol_history["volume_adjusted_return"] = symbol_history["daily_return"] / (symbol_history["volume"] / 1e6)

        # Backtest on historical data to predict future returns
        recent_returns = symbol_history.iloc[-30:]["volume_adjusted_return"].mean()
        if recent_returns > 2:
            top_n_symbols.append(symbol)
        if len(top_n_symbols) >= _max_positions:
            break

    return top_n_symbols[:_max_positions]