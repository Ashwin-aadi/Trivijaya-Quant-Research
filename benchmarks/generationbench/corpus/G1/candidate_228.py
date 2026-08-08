from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves buying stocks that have outperformed the market "
        "in recent periods. This strategy leverages the tendency for past winners to continue "
        "performing well."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate returns
        returns: pl.DataFrame = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
            .collect()
        )

        # Filter out symbols without sufficient data
        if returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s in closes.columns if s in returns.select(pl.col("symbol"))]

        # Sort by average return and select top N
        ranked_returns = (
            returns.sort("avg_return", descending=True)
            .select(["symbol"])
            .to_pandas()
            .head(self._top_n)["symbol"]
            .tolist()
        )

        if not ranked_returns:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_returns)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_returns},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest