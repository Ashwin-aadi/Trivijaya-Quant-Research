from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Higher liquidity stocks tend to have lower bid-ask spreads and higher trading volumes, "
        "potentially leading to more efficient price discovery and reduced transaction costs. "
        "By equal-weighting these stocks, we aim to benefit from their relative stability."
    )

    def __init__(self, lookback: int = 30) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_ratios = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            volume_ratio = (
                float(history[symbol]["volume"].sum())
                / float(view.latest_close()[symbol])
            )
            volume_ratios[symbol] = volume_ratio

        ranked_symbols = sorted(volume_ratios.keys(), key=lambda s: volume_ratios[s], reverse=True)
        top_symbols = ranked_symbols[: min(len(ranked_symbols), 10)]
        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(
            information_available_at=stamp, weights={s: weights.get(s, 0.0) for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest