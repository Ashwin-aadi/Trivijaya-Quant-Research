from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically shown higher returns than high-volatility "
        "stocks. This phenomenon is known as the low-volatility anomaly and can be exploited "
        "by tilting a portfolio towards lower volatility assets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            log_returns = [
                (math.log(c / prev_c)) * 100.0 for c, prev_c in zip(adj_closes[1:], adj_closes[:-1])
            ]
            volatility = math.sqrt(sum([x**2 for x in log_returns]) / len(log_returns))
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        picks = sorted_symbols[:5]
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