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
        closes = view.closes(lookback=120)

        if closes.height < 120:
            return Signal(information_available_at=stamp, weights={})

        seasonal_symbols = set()
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < 120:
                continue

            month_indices = [int(date.fromtimestamp(t).strftime('%m')) - 1 for t in values[:-1]]
            season_occurrences = sum(1 for i, month in enumerate(month_indices) if str(i+1) == self._season)
            if season_occurrences >= 2:
                seasonal_symbols.add(symbol)

        if not seasonal_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(seasonal_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in seasonal_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest