from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalBreakout(Strategy):
    rationale = (
        "Certain stocks in India exhibit stronger performance during specific seasons or holidays. "
        "For instance, agricultural-related stocks might see increased trading volumes and prices during harvest periods. "
        "Identifying these patterns can provide a basis for timing entries into the market."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Define the holiday season as the last 30 days of November and first 15 days of January
        holiday_season = {str(date(y, 11, 25)) for y in range(2020, 2024)} | {
            str(d) for d in [date(y, 1, m) for y in range(2020, 2024) for m in range(1, 16)]
        }

        # Filter the close prices to only include those within the holiday season
        relevant_closes = closes.filter(pl.col("session_date").is_in(holiday_season))
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in relevant_closes.columns:
                continue
            
            values = [float(v) for v in relevant_closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if values[-1] >= max(values):
                picks.append(symbol)

        picks = picks[: self._top_n]
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