from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where low-volatility stocks tend to outperform "
        "high-volatility stocks over time. By systematically tilting towards less volatile stocks, "
        "the portfolio aims to capture higher returns associated with lower risk."
    )

    def __init__(self, window: int = 20, bottom_n: int = 30) -> None:
        self._window = window
        self._bottom_n = bottom_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        history = history.select(["session_date", "symbol"] + symbols)

        returns = (history
                   .with_columns([
                       ((pl.col(col) - pl.col(col).shift(1)) / pl.col(col).shift(1)).alias(f"r_{col}")
                       for col in symbols
                   ])
                   .drop_nulls()
                   .sort("session_date")
                   .tail(self._window)
                   .select(["symbol"] + [f"r_{col}" for col in symbols])
                  )

        if returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = (returns
                      .with_columns([
                          (pl.col(f"r_{sym}").std().alias(f"vol_{sym}"))
                          for sym in symbols
                      ])
                      .select(["symbol"] + [f"vol_{sym}" for sym in symbols])
                     )

        lowest_volatility_symbols = volatility.sort(pl.col("vol_0"), descending=False).head(self._bottom_n)["symbol"].to_list()

        weight = 1.0 / len(lowest_volatility_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in lowest_volatility_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest