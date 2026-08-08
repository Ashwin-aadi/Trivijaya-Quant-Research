from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Highly liquid stocks are more likely to be fairly priced and react quickly to news. "
        "By investing in the most liquid names, we aim to reduce transaction costs and improve "
        "the likelihood of capturing market movements."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_sum = (
            history.groupby("symbol")
                   .agg(pl.col("volume").sum())
                   .sort("volume", descending=True)
                   .select(["symbol"])
        )
        symbols = [row["symbol"] for row in volume_sum.to_dicts()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

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