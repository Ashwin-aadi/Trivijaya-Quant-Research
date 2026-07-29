from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks in the NIFTY 100 may exhibit stronger performance during specific "
        "calendar periods due to seasonal effects or event-driven factors. This strategy aims "
        "to capitalize on these patterns by identifying stocks that historically perform well "
        "during certain months."
    )

    def __init__(self, season: str = "October") -> None:
        self._season = season

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365 * 5)  # Look back over the past 5 years
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        season_mask = (history["session_date"].dt.month() == _month_from_name(self._season))
        seasonal_history = history.filter(season_mask)
        if seasonal_history.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        returns = (
            seasonal_history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by(["symbol"])
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
        )

        top_symbols = [row["symbol"] for row in returns.to_dicts()[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def _month_from_name(season: str) -> int:
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    return month_map.get(season, 1)