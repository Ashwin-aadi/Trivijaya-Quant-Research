from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy exploits price-level reversion by identifying stocks that have "
        "deviated from their long-term average prices. It aims to enter positions in "
        "undervalued stocks and exit when prices revert."
    )

    def __init__(self, window: int = 50, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_ranks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            ma_values = _rolling_mean(values, self._window)
            current_price = values[-1]
            deviation = (current_price - ma_values[-1]) / ma_values[-1]
            if deviation < -0.1:
                rank = 1 + abs(deviation) * 10
                symbol_ranks[symbol] = rank

        top_n_symbols = sorted(symbol_ranks, key=lambda k: symbol_ranks[k], reverse=True)[:20]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rolling_mean(values: list[float], window: int) -> list[float]:
    n = len(values)
    if n < window:
        return [sum(values) / n] * (n + 1 - window)

    cumsum = [0.0]
    for i, value in enumerate(values):
        cumsum.append(cumsum[i] + value)

    rolling_sum = [(cumsum[j] - cumsum[max(0, j - window)]) for j in range(n)]
    return [s / window for s in rolling_sum[window - 1:]]