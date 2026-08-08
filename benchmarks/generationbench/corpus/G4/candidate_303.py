from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy leverages the anomaly where less-liquid stocks tend to underperform "
        "more liquid ones due to trading frictions and information asymmetry. By equally "
        "weighting highly liquid stocks selected from the top decile, we aim to capture outperformance "
        "from relatively neglected but investable illiquid stocks while maintaining diversification."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        liquidity_scores = pl.DataFrame({"symbol": symbols})
        for symbol in symbols:
            liquidity_score = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    (pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1.0
                )
                .sort("session_date", descending=True)
                .head(self._window)
                .select(pl.col("adj_close").mean())
            )
            liquidity_scores = liquidity_scores.with_column(liquidity_score.alias(symbol))

        top_decile_symbols = [
            symbol for _, symbol in liquidity_scores.sort(symbol, descending=True).rows()[: self._top_n]
        ]
        weight = 1.0 / len(top_decile_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_decile_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest