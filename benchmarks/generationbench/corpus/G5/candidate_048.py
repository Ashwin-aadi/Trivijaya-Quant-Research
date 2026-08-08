from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects the top performers across a cross-section of symbols over a "
        "lookback period. The idea is that stocks with strong recent performance are more "
        "likely to continue outperforming due to momentum effects."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol over the window period
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window - 1) - 1.0).alias("return")
            )
            .drop_nulls()
        )

        # Sort by return in descending order to get the top performers
        sorted_history = history.sort("return", descending=True)
        symbols_with_high_momentum: list[str] = [row["symbol"] for row in sorted_history.select("symbol").to_dicts()[:self._top_n]]

        if not symbols_with_high_momentum:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_high_momentum)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_high_momentum}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest