from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit seasonal behavior due to industry-specific factors or calendar effects. "
        "By identifying these patterns, we can capitalize on predictable price movements."
    )

    def __init__(self, window: int = 60, threshold: float = 0.15) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            returns = [(adj_closes[i] - adj_closes[i-1]) / adj_closes[i-1] for i in range(1, len(adj_closes))]
            seasonal_effect = sum(returns[-3:]) > self._threshold
            symbol_data[symbol] = (seasonal_effect, adj_closes[-1])

        picks: list[str] = [symbol for symbol, data in symbol_data.items() if data[0]]
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