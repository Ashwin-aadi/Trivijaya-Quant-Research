from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy equal weights a selection of stocks based on their liquidity. "
        "Highly liquid stocks are more likely to be stable and provide better execution."
    )

    def __init__(self, min_volume: int = 100000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter stocks by minimum volume
        filtered_history = (
            history.filter(
                (pl.col("volume") > self._min_volume) & (pl.col("symbol").is_not_null())
            )
        )

        if filtered_history.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight selection
        symbols = filtered_history["symbol"].unique().to_list()
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest