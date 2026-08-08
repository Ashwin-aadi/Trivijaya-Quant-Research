from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Equities often exhibit seasonality, with certain periods of the year showing stronger "
        "returns than others due to economic cycles or market psychology. This strategy aims to "
        "capitalize on these seasonal trends by identifying symbols that historically perform well "
        "during specific times of the year."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factor: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            mean_close = sum(closes[-self._window:]) / self._window

            # Calculate the average return over the last `window` days
            returns = [(closes[i] - closes[i - 1]) / closes[i - 1] if i > 0 else 0 for i in range(len(closes))]
            mean_return = sum(returns[-self._window:]) / self._window

            # Seasonality factor is the current return relative to the average return
            current_return = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
            seasonality_factor[symbol] = current_return / mean_return if mean_return != 0 else 0

        sorted_factors = sorted(seasonality_factor.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_factors][:3]

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