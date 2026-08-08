from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in India exhibit stronger performance during specific months of the year "
        "due to seasonal trends or calendar effects. Identifying these patterns can provide "
        "predictive signals for profitable trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        seasonal_trends: dict[str, float] = {}

        for symbol in symbols:
            closes = history[symbol].to_list()
            monthly_closes = [
                sum(closes[i * 21 : (i + 1) * 21]) / min(21, len(closes) - i * 21)
                for i in range(self._window // 21 + 1)
            ]
            if not monthly_closes:
                continue
            strongest_month = monthly_closes.index(max(monthly_closes))
            seasonal_trends[symbol] = (strongest_month + 1) / self._window

        sorted_symbols = [
            s for _, s in sorted(seasonal_trends.items(), key=lambda item: -item[1])
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest