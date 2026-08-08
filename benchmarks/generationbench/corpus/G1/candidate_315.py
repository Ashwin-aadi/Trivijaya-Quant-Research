from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening filters out low-liquidity stocks before equal-weighting the "
        "remaining universe. This strategy aims to balance risk and reward by favoring more "
        "liquid stocks, which are easier to trade without significantly moving the market."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.lazy()
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .with_columns(
                (pl.col("avg_volume") / pl.col("avg_volume").max() * 100).alias("liquidity_score")
            )
            .collect()["liquidity_score"]
            .to_list()
        )

        filtered_symbols = [
            symbol for symbol, score in zip(view.symbols, liquidity_scores) if score > 0
        ]

        weights = {symbol: 1.0 / len(filtered_symbols) for symbol in filtered_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest