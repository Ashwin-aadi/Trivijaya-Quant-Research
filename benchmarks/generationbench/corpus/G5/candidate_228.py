from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks which have performed well "
        "relative to the market over a lookback period are likely to continue outperforming. "
        "This strategy allocates capital to the top performers based on their relative performance."
    )

    def __init__(self, window: int = 20, num_top_symbols: int = 5) -> None:
        self._window = window
        self._num_top_symbols = num_top_symbols

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.closes().select("adj_close").to_dict()[1][0]
        returns = (
            closes
            .lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .sort(by="avg_return", descending=True)
            .collect()
        )

        top_symbols = [row["symbol"] for row in returns.to_dicts()[:self._num_top_symbols]]
        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, pl.Date)
    return newest