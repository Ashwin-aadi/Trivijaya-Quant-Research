from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well over a recent period to continue performing well. This is based on the "
        "assumption that strong performers are likely to remain strong."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns over the lookback period
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .select(["symbol", "session_date", "return"])
        )

        # Get the latest close and return
        latest_closes = view.closes(lookback=None)

        # Filter symbols that are in both history and latest closes
        common_symbols = set(history_with_returns["symbol"]) & set(latest_closes.columns)
        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})

        recent_performers = (
            history_with_returns.filter(pl.col("return").is_not_null())
                                 .filter(pl.col("symbol").is_in(common_symbols))
                                 .group_by("symbol")
                                 .agg(
                                     pl.col("return").mean().alias("avg_return"),
                                 )
                                 .sort("avg_return", descending=True)
        )

        top_n_performers = recent_performers.head(self._window)["symbol"].to_list()[:5]

        if not top_n_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest