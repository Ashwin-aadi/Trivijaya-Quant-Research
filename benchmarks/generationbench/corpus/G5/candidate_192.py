from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the fact that stocks with positive "
        "returns over a recent period are likely to continue outperforming in the near future."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = (
            history.select(
                pl.col("symbol"),
                (pl.col("close") / pl.col("open").shift(self._window) - 1.0).alias("momentum")
            )
            .sort("momentum", descending=True)
            .group_by("symbol")
            .agg(pl.col("momentum").mean().alias("average_momentum"))
        )

        top_symbols = momentum_scores.sort("average_momentum", descending=True).select(
            pl.col("symbol").head(self._top_n)
        ).to_dict(as_series=False)["symbol"]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_dict(as_series=False)["session_date"]
    assert isinstance(newest, date)
    return newest