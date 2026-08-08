from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening aims to identify stocks with high single-day trading volume. "
        "These stocks are more likely to be available at the desired price and quantity, "
        "reducing slippage risk. Equal weighting across selected assets promotes a balanced portfolio."
    )

    def __init__(self, liquidity_threshold: float = 50_000) -> None:
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.select(pl.col("symbol"), pl.col("volume").alias("daily_volume"))
            .filter(pl.col("daily_volume") > self._liquidity_threshold)
            .select("symbol")
            .to_dict(True)["symbol"]
        )

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volume_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date), "Expected session_date to be of type date"
    return newest