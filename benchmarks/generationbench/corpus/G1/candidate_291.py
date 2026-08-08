from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening helps identify stocks with lower trading costs and higher "
        "marketability. Equal weighting across these stocks can provide a diversified portfolio."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return_ratio")
            )
            .sort("total_volume", descending=True)
            .head(20)
        )

        if liquidity_screened.height < 20:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in liquidity_screened.to_dict(as_series=False).values()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest