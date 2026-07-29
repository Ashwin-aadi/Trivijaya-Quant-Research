from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening helps to identify stocks that are easier to trade without impacting "
        "the market price significantly. Equal weighting ensures each selected stock contributes"
        "equally to the portfolio, providing a balanced exposure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        liquidity_scores = (
            history.group_by("symbol")
                   .agg((pl.col("volume").sum() / 20).alias("liquidity_score"))
        )

        top_symbols = liquidity_scores.sort("liquidity_score", descending=True)
        if top_symbols.height < len(symbols):
            return Signal(information_available_at=stamp, weights={})

        top_symbols_list = [str(symbol) for symbol in top_symbols["symbol"].to_list()[:len(symbols)]]
        weight = 1.0 / len(top_symbols_list)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols_list}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest