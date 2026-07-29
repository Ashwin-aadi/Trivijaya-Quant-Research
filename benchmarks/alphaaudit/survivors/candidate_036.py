from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that outperformed in the recent past "
        "to continue outperforming. This strategy ranks stocks based on their returns over a lookback period and "
        "buys the top-performing stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
            .drop("open", "high", "low", "close", "volume")
        )

        # Group by symbol and calculate the mean return over the window
        mean_returns = history.group_by("symbol").agg(
            (pl.col("return").mean().alias("avg_return"))
        ).sort("avg_return", descending=True)

        if mean_returns.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in mean_returns.to_dicts()[:5]]
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