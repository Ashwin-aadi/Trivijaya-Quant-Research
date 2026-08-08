from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the anomaly where small-cap stocks often outperform large caps in the Indian market. By using a liquidity screen to filter stocks with high trading volume and equal-weighting them, we aim to capture this effect without overconcentration in any single stock. The persistence of this phenomenon may stem from asymmetric information and institutional trading biases."
    )

    def __init__(self, top_n: int = 30, lookback_days: int = 30) -> None:
        self._top_n = top_n
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_data = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
            )
            .sort("avg_volume", descending=True)
            .head(self._top_n)
        )

        if liquidity_data.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in liquidity_data.iter_rows()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest