from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits the tendency for stocks with strong past performance (momentum) "
        "to continue outperforming in the near future. By ranking stocks based on their cumulative "
        "returns over a lookback period and allocating capital to top-performing ones, we aim to "
        "capitalize on this momentum effect while implementing risk management techniques such as "
        "position sizing and stop-loss orders."
    )

    def __init__(self, window: int = 60, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        closes = view.closes(lookback=self._window)
        returns: pl.DataFrame = (
            history.select(
                "symbol",
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"),
            )
            .with_columns(pl.col("return").cumsum().alias("cum_return"))
            .filter(pl.col("session_date") == history["session_date"].max())
            .select(["symbol", "cum_return"])
        )

        ranked_symbols = (
            returns.group_by("symbol")
            .agg(
                pl.col("cum_return").mean().alias("avg_cum_return"),
            )
            .sort("avg_cum_return", descending=True)
            .head(self._top_n)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest