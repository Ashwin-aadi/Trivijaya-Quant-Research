from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume "
        "are considered. Equal weighting across these stocks can provide a balanced approach "
        "in capturing market movements without the bias towards larger-cap stocks."
    )

    def __init__(self, min_volume: int = 100_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by minimum volume
        filtered_history = (
            history.filter(
                (pl.col("volume") > self._min_volume) & (pl.col("session_date").is_not_null())
            )
            .sort("session_date", descending=True)
            .head(365)
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Equal weight across the screened symbols
        symbols = [row["symbol"] for row in filtered_history.to_dicts()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest