from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well recently to continue outperforming those that have lagged. By allocating more "
        "weight to top performers and less to laggards, this strategy aims to capture "
        "positive momentum effects."
    )

    def __init__(self, window: int = 20, num_top_assets: int = 5) -> None:
        self._window = window
        self._num_top_assets = num_top_assets

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (
            closes
            .lazy()
            .with_column((pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r"))
            .collect()
        )

        # Select top performing assets
        ranked_returns = returns.sort("r", descending=True)
        picks: list[str] = [row[0] for row in ranked_returns["symbol"].to_list()[: self._num_top_assets]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to top assets
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