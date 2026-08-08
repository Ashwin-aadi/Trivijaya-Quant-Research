from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy exploits seasonal effects in the Indian market by identifying "
        "historically strong performing sectors during specific months. By focusing on "
        "these patterns, we aim to generate profitable trades while managing risk through "
        "diversification and position sizing."
    )

    def __init__(self, lookback_period: int = 10, top_n_sectors: int = 3) -> None:
        self._lookback_period = lookback_period
        self._top_n_sectors = top_n_sectors

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period + 1)

        if history.height < self._lookback_period + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate average monthly returns
        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            closes = history.select(pl.col(symbol)).to_series()
            month_ends = [date(closes[i].year, (i + 1) // 3 * 3 + 2, 1) for i in range(0, len(closes), 3)]
            monthly_closes = [closes[i - 1] for i in range(len(month_ends)) if i < len(closes)]
            avg_return = sum((monthly_closes[i].close / monthly_closes[i - 1].close - 1.0) for i in range(1, len(monthly_closes))) / (len(monthly_closes) - 1)
            avg_returns[symbol] = avg_return

        # Rank sectors based on average returns
        ranked_sectors = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)[:self._top_n_sectors]

        if not ranked_sectors:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n_sectors
        sector_weights = {s: weight for s, _ in ranked_sectors}
        return Signal(
            information_available_at=stamp, weights=sector_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest