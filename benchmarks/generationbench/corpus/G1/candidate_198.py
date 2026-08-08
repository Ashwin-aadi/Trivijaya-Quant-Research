from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength strategy selects the top N performing stocks based on their "
        "performance compared to the broader market index. This assumes that outperforming "
        "stocks are more likely to continue their positive momentum."
    )

    def __init__(self, window: int = 20, n_assets: int = 5) -> None:
        self._window = window
        self._n_assets = n_assets

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            avg_price = sum(prices[-10:]) / 10.0  # Use a rolling average to smooth data
            strength = max(prices) / avg_price - 1.0
            relative_strengths[symbol] = strength

        sorted_strengths = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_strengths[: self._n_assets]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest