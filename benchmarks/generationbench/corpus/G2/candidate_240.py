from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the past to continue outperforming in the future. By ranking stocks based on recent "
        "returns and selecting top performers, we can capture this effect."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Compute returns
        returns = (
            history.lazy()
            .with_columns(
                (pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.sum("return").alias("total_return"))
            .collect()
        )

        # Rank symbols by total return
        ranked_symbols = returns.sort("total_return", descending=True)

        picks: list[str] = [row["symbol"] for _, row in ranked_symbols.iter_rows()][: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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