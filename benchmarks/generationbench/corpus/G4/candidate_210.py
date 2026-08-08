from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where stocks with strong relative performance "
        "against the broad universe often continue to outperform. By focusing on top N stocks "
        "based on their relative returns over a lookback period, we aim to capture residual "
        "performance and reduce market-wide risks."
    )

    def __init__(self, window: int = 90, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        index_symbols = view.symbols
        stock_history = history[index_symbols]
        index_data = stock_history.filter(pl.col("symbol").is_in(index_symbols))

        returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in index_data.columns:
                continue
            prices = [float(v) for v in index_data[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            cumulative_return = (prices[-1] - prices[0]) / prices[0]
            returns[symbol] = cumulative_return

        ranked_symbols = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)[: self._top_n]
        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest