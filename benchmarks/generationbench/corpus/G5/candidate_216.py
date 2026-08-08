from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for recent winners (in terms of returns) "
        "to continue outperforming in the near future. This strategy allocates capital to top "
        "performers over a lookback period."
    )

    def __init__(self, window: int = 20, num_top_performers: int | None = None) -> None:
        self._window = window
        self._num_top_performers = num_top_performers or len(self.view.symbols)

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close"))
        returns = (
            (closes.shift(-1) / closes - 1.0).select(pl.all().alias("r"))
        )

        top_returns: pl.DataFrame | None = None
        if not returns.is_empty():
            top_returns = (
                returns.sort("r", descending=True)
                .group_by("symbol")
                .agg(pl.col("r").mean().alias("mean_return"))
            )

        top_performers: list[str] = []
        for symbol in view.symbols:
            if symbol not in top_returns.columns and not top_returns.is_empty():
                continue
            mean_return = float(top_returns.filter(pl.col("symbol") == symbol)["mean_return"].item())
            if mean_return >= returns["r"].quantile(0.9):
                top_performers.append(symbol)

        top_performers = top_performers[: self._num_top_performers]
        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest