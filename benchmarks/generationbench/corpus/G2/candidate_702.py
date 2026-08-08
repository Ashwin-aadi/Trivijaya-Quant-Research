from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed the broader market in recent periods are more likely "
        "to continue their outperformance due to momentum effects. This strategy aims to "
        "identify and invest in such stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        nifty_close = closes.select(
            pl.col("NIFTY_100").alias("nifty")
        ).to_dict(False)["nifty"]
        symbols = [symbol for symbol in view.symbols if symbol != "NIFTY_100"]

        returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("return", descending=True)
            .collect()
        )

        top_n_symbols = [row["symbol"] for row in returns.to_dict(False)[: self._top_n]]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest