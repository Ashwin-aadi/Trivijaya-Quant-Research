from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are often more efficient in price discovery and trading. "
        "By focusing on a subset of highly liquid stocks, we may capture the benefits of better "
        "price formation without the high bid-ask spread costs that can disproportionately affect "
        "less liquid names."
    )

    def __init__(self, liquidity_threshold: float = 10_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter for the highest liquidity stocks
        liquidity_filtered = (
            history.filter(pl.col("volume") > self._threshold).group_by("symbol").agg(
                pl.count().alias("count")
            )
        ).sort("count", descending=True)

        if liquidity_filtered.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in liquidity_filtered.head(5).rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest