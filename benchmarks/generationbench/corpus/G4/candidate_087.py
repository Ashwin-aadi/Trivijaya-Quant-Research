from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the persistence in stock returns through a liquidity-screened equal-weighting approach. By focusing on stocks with higher trading volumes, it mitigates liquidity risk while ensuring each selected stock receives an equal weight in the portfolio."
    )

    def __init__(self, lookback_days: int = 30, top_percentile: float = 0.1) -> None:
        self._lookback_days = lookback_days
        self._top_percentile = top_percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute average daily trading volume over the lookback period
        avg_volume = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume") / self._lookback_days).alias("avg_volume")
            )
            .group_by("symbol")
            .agg(pl.col("avg_volume").mean().alias("m"))
            .sort("m", descending=True)
        )

        # Select top percentile of symbols based on average volume
        n_symbols = avg_volume.height
        cutoff = int(n_symbols * self._top_percentile)
        selected_symbols = [row[0] for row in avg_volume.head(cutoff).to_dict(as_series=False)["symbol"]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting among the selected symbols
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest