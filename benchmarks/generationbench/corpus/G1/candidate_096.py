from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue performing well. This strategy ranks symbols based on "
        "recent returns and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate returns
        history = history.with_columns(
            (pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r")
        )

        # Get the latest close prices for ranking
        closes["r"] = [
            float(history.select(pl.col(symbol)).to_list()[0][-1])
            if symbol in history.columns else 0.0
            for symbol in closes.columns[:-1]
        ]

        # Rank symbols by return
        ranks = (
            closes.sort("session_date", descending=True)
            .select([pl.last("r").alias(f"symbol_{i}") for i in range(self._top_n)])
            .to_numpy()
        )

        top_symbols: list[str] = []
        if not ranks.any(axis=1).any():
            return Signal(information_available_at=stamp, weights={})

        # Get the top N symbols
        for symbol_rank in ranks:
            if symbol_rank[0]:
                top_symbols.append(closes.columns[symbol_rank[0]])
                break

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