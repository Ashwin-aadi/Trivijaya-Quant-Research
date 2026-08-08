from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that assets which have outperformed their "
        "peers over a recent period are more likely to continue outperforming in the near term."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Compute returns for each symbol
        returns = (
            closes.melt().with_columns(
                (pl.col("value") / pl.col("value").shift(self._window) - 1).alias("return")
            )
        ).filter(pl.col("variable").is_not_null())

        top_performers = (
            returns.groupby("variable")
            .agg((pl.col("return").mean().alias("avg_return")))
            .sort("avg_return", descending=True)
            .head(self._top_n)
        )["variable"].to_list()

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest