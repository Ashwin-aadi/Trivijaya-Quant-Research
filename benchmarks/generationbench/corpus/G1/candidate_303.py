from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for stocks with high liquidity before equally weighting them. "
        "High liquidity ensures that trades can be executed without significantly affecting the price. "
        "The equal weighting across selected assets aims to provide a balanced exposure."
    )

    def __init__(self, min_volume: float = 10_000_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_filter = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume").rolling_sum(window_size=20) / 20).alias("avg_volume")
            )
            .filter(pl.col("avg_volume") > self._min_volume)
            .select("symbol")
            .to_dict(as_series=False)["symbol"]
        )

        if not volume_filter:
            return Signal(information_available_at=stamp, weights={})

        filtered_symbols = [s for s in view.symbols if s in volume_filter]
        equal_weight = 1.0 / len(filtered_symbols)
        weights = {s: equal_weight for s in filtered_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest