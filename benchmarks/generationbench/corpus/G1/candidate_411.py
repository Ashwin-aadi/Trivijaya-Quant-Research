from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and can indicate the ease with which an asset "
        "can be traded without affecting its price. This strategy screens for high liquidity "
        "assets to ensure that trades do not significantly impact the stock price, leading to "
        "more stable execution."
    )

    def __init__(self, min_volume: int = 1000000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Look at the last year for volume
        if history.is_empty() or history.height < 252:
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = (
            history.select(["symbol", "volume"])
            .filter(pl.col("volume") >= self._min_volume)
            .select("symbol")
            .unique()
            .to_dict(as_series=False)["symbol"]
        )

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(high_volume_symbols)
        weights = {s: equal_weight for s in high_volume_symbols}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest