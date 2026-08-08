from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening can help identify stocks that are more actively traded and may "
        "provide better price discoverability. Equal weighting across a subset of highly liquid "
        "stocks could lead to a strategy that performs well in terms of risk-adjusted returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
                (pl.col("close") - pl.col("open")).abs().mean().alias("price_range"),
            )
            .with_columns(
                ((pl.col("avg_volume") / pl.col("avg_volume").max()) * 10).round(2)
                .alias("liquidity_score")
            )
            .sort("liquidity_score", descending=True)
            .select("symbol", "liquidity_score")
        )

        liquidity_top_symbols = [row["symbol"] for row in liquidity_scores.to_dicts()][:5]

        if not liquidity_top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest