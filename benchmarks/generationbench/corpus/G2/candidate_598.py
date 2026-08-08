from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Certain sectors or industries exhibit seasonality in their performance. For instance,"
        "consumer discretionary stocks may experience higher returns during festive seasons."
        "Identifying such patterns can lead to profitable trading opportunities."
    )

    def __init__(self, window: int = 30, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            last_close = values[-1]
            mean_close = sum(values[-self._threshold :]) / min(
                self._threshold, len(values)
            )
            if last_close > mean_close * 1.05:  # Increase the threshold
                seasonal_symbols.append(symbol)

        seasonal_symbols = seasonal_symbols[:20]  # Limit to top 20 symbols
        if not seasonal_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(seasonal_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in seasonal_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest