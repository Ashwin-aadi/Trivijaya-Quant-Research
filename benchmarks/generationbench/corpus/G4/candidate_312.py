from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum3M(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying and investing in "
        "stocks that have recently outperformed their peers over a 3-month period. The mechanism "
        "is based on the persistence of strong performance, driven by positive investor sentiment "
        "and company-specific events."
    )

    def __init__(self, lookback_days: int = 90, long_positions: int = 10, short_positions: int = 10) -> None:
        self._lookback_days = lookback_days
        self._long_positions = long_positions
        self._short_positions = short_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = self._calculate_cumulative_returns(history)
        picks_long, picks_short = self._select_top_bottom_stocks(cumulative_returns)

        weight_long = 1.0 / len(picks_long) if picks_long else 0
        weight_short = -1.0 / len(picks_short) if picks_short else 0

        weights = {symbol: weight_long for symbol in picks_long}
        weights.update({symbol: weight_short for symbol in picks_short})

        return Signal(
            information_available_at=stamp, weights={s: float(w) for s, w in weights.items()}
        )

    def _calculate_cumulative_returns(self, history: pl.DataFrame) -> pl.DataFrame:
        symbols = view.symbols
        closes = history.select(pl.col("symbol").is_in(symbols)).select(
            "symbol", (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_days) - 1.0).alias("cumulative_return")
        )
        return closes

    def _select_top_bottom_stocks(self, cumulative_returns: pl.DataFrame) -> tuple[list[str], list[str]]:
        top_n = self._long_positions
        bottom_m = self._short_positions

        sorted_by_return = cumulative_returns.sort("cumulative_return", descending=True).head(top_n + bottom_m)
        picks_long = [row["symbol"] for row in sorted_by_return.to_dicts()[:top_n]]
        picks_short = [row["symbol"] for row in sorted_by_return.to_dicts()[top_n:]]
        
        return picks_long, picks_short


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest