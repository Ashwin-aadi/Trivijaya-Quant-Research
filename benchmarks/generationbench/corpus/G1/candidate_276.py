from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with the highest recent returns "
        "are likely to continue outperforming. This strategy aims to identify such "
        "outperformers and allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes.sort("session_date")
            .select(
                pl.col("adj_close").shift(-self._window).alias(f"close_{self._window}"),
                pl.col("adj_close").alias("close_now"),
            )
            .with_column((pl.col("close_now") / pl.col(f"close_{self._window}") - 1.0).alias("return"))
        )

        top_symbols = returns.sort("return", descending=True).select(
            "symbol"
        ).to_series().to_list()[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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