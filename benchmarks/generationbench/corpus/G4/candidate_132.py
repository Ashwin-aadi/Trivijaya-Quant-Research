from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks that have "
        "historically outperformed their peers over a specific lookback period. Strong performers "
        "are selected for long positions, while underperformers are shorted, leveraging the "
        "tendency of momentum to persist in equity markets."
    )

    def __init__(self, window: int = 60, top_decile_size: int = 20) -> None:
        self._window = window
        self._top_decile_size = top_decile_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) < 2 * self._top_decile_size:
            return Signal(information_available_at=stamp, weights={})

        # Calculate average monthly returns for each stock
        avg_returns = (
            closes[[f"close_{i}" for i in range(self._window)]]
            .transpose()
            .with_columns(
                (pl.col("column") / pl.col("column").shift(1) - 1.0).alias("monthly_return")
            )
            .drop(["column"])
            .group_by("symbol")
            .agg(pl.col("monthly_return").mean().alias("avg_monthly_return"))
            .sort("avg_monthly_return", descending=True)
        )

        # Rank stocks based on average monthly returns
        rank = avg_returns.with_column(
            pl.col("avg_monthly_return").rank(method="ordinal", descending=True).alias("rank")
        )
        top_decile_symbols = [r[0] for r in rank.head(self._top_decile_size).to_dicts()]
        bottom_decile_symbols = [
            r[0]
            for r in rank.tail(self._top_decile_size)
            .reverse()
            .head(self._top_decile_size)
            .to_dicts()
        ]

        # Assign weights
        top_weights = {s: 1.0 / self._top_decile_size for s in top_decile_symbols}
        bottom_weights = {s: -1.0 / self._top_decile_size for s in bottom_decile_symbols}

        return Signal(
            information_available_at=stamp,
            weights={
                **top_weights,
                **bottom_weights,
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest