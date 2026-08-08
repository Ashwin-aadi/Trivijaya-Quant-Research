from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and information flow. Highly liquid stocks "
        "are less likely to be manipulated or overvalued due to insufficient demand. By equal "
        "weighting the most liquid stocks, we aim to benefit from this robustness."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_metrics = [
            float(v) for v in history.select(pl.col("volume").sum()).to_series().to_list()[0]
        ]
        top_n_symbols = sorted(
            view.symbols,
            key=lambda symbol: liquidity_metrics[history[symbol]["volume"].argmax()],
            reverse=True,
        )[: self._window]

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