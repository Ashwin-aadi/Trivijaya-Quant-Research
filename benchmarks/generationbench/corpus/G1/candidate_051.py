from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity plays a crucial role in market efficiency. Equally weighting stocks based on their "
        "liquidity can help identify more liquid assets which are generally better for trading."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .with_columns(pl.col("total_volume").rank(method="dense", descending=True).alias("liquidity_score"))
        )

        if liquidity_scores.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbol_scores = [float(v) for v in liquidity_scores["liquidity_score"].to_list()]
        symbols = [s.strip() for s in history["symbol"].to_list()]

        top_symbols = sorted(zip(symbol_scores, symbols), reverse=True)[:5]
        top_symbols_dict = {s: 1.0 / len(top_symbols) for _, s in top_symbols}

        return Signal(information_available_at=stamp, weights=top_symbols_dict)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest