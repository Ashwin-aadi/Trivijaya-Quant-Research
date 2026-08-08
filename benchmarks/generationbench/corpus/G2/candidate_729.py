from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often less prone to extreme price movements and may offer "
        "more stable returns over time. By tilting our portfolio towards these stocks, we aim "
        "to capture the risk premium associated with lower volatility."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close").std() / pl.col("adj_close").mean()).alias("volatility")
            )
            .group_by("symbol", maintain_order=True)
            .agg(pl.col("volatility").min().alias("min_volatility"))
        )

        if volatility.is_empty():
            return Signal(information_available_at=stamp, weights={})

        min_volatility = float(volatility["min_volatility"].max())
        top_symbols = volatility.filter(
            (pl.col("min_volatility") == min_volatility) & pl.col("symbol").is_in(view.symbols)
        )["symbol"].to_list()

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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest