from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "High liquidity in stocks is crucial for minimizing trading costs and executing trades efficiently. "
        "This strategy allocates weights based on a combination of volume and price range metrics to identify "
        "highly liquid stocks that can be traded with minimal impact."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean().cast(pl.float64)).alias("avg_vol"),
                (pl.col("adj_close") - pl.col("open")).abs().mean().alias("price_range"),
            )
            .sort(by="avg_vol", descending=True)
        )

        if liquidity_scores.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in liquidity_scores.to_dicts()][: self._window]
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