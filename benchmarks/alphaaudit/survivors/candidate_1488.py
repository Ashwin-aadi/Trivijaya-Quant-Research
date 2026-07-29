from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue outperforming the market. This strategy ranks assets by "
        "recent returns and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbols = [symbol for symbol in view.symbols if symbol in latest_closes.columns]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("return"))
        )

        # Rank by return
        ranked = history.sort("return", descending=True)

        top_n_symbols = symbols[: self._window]
        weights: dict[str, float] = {s: 1.0 / len(top_n_symbols) for s in top_n_symbols}
        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest