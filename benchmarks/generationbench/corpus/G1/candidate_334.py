from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Stocks often exhibit seasonal patterns due to macroeconomic factors, "
        "such as weather-related events or holiday seasonality. This strategy aims to identify "
        "and exploit these patterns by focusing on periods with historically positive returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_effect: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) < self._window:
                continue
            returns = [
                (float(history[symbol][i + 1]) - float(history[symbol][i])) / float(history[symbol][i])
                for i in range(len(history[symbol]) - 1)
            ]
            avg_return = sum(returns) / len(returns)
            if avg_return > 0:
                seasonal_effect[symbol] = avg_return

        sorted_symbols = [
            s
            for _, s in sorted(seasonal_effect.items(), key=lambda item: item[1], reverse=True)
        ][:5]

        weights = {s: 1.0 / len(sorted_symbols) for s in sorted_symbols}
        return Signal(
            information_available_at=stamp, weights={**weights, "CASH": 1 - sum(weights.values())}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest