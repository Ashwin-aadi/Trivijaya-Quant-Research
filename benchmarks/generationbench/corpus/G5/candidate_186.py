from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks with higher relative strength compared to the broader market "
        "can provide a potential edge. This strategy focuses on selecting the top-performing "
        "stocks over a given period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = [s for s in view.symbols if s in closes.columns]
        symbol_ranks = (
            history.filter(pl.col("symbol").is_in(symbols))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return_ratio"),
            )
            .sort(by="return_ratio", descending=True)
        )

        if symbol_ranks.height < len(symbols):
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in symbol_ranks.head(self._top_n).to_dicts()]
        weight = 1.0 / max(len(top_symbols), 1)  # Ensure at least one stock gets a non-zero weight
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