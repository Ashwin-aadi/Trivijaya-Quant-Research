from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Firms with strong relative strength are likely to continue outperforming their "
        "peers. This strategy identifies the top performers based on their historical returns "
        "relative to the broader market and allocates capital accordingly."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history["symbol"].unique()) < 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Compute average daily returns for each symbol
        avg_returns = (
            history.group_by("symbol")
            .agg(
                pl.col("r").mean().alias("avg_return"),
            )
            .collect()
        ).to_pandas()

        top_symbols = avg_returns.sort_values(by="avg_return", ascending=False).head(5)["symbol"].tolist()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest