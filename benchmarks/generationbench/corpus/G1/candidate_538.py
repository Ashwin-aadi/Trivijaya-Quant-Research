from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Some stocks exhibit stronger performance during specific months or seasons of the year. "
        "This strategy aims to identify such seasonal effects and allocate capital accordingly."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            monthly_closes = [
                sum(values[i : i + 20]) / 20.0 for i in range(0, len(values), 20)
            ]
            max_monthly_close = max(monthly_closes)

            seasonal_factors[symbol] = (
                values[-1] / max_monthly_close if max_monthly_close != 0 else 0
            )

        top_symbols = sorted(
            seasonal_factors.items(), key=lambda item: -item[1]
        )[: self._top_n]

        weights = {s: 1.0 / len(top_symbols) for s, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest