from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and accessibility. "
        "By focusing on the most liquid stocks, we aim to reduce trading costs and improve execution quality."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols with insufficient trading volume
        min_volume = 10_000_000  # Example threshold for minimum daily trading volume
        filtered_history = (
            history.filter(
                pl.col("volume") > min_volume,
                pl.col("symbol").is_in(view.symbols),
            )
            .group_by("symbol")
            .agg(
                (pl.col("volume").mean().alias("avg_vol")),
            )
            .sort(pl.col("avg_vol"), descending=True)
        )

        if filtered_history.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting among the most liquid symbols
        top_symbols = [row["symbol"] for row in filtered_history.to_dicts()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest