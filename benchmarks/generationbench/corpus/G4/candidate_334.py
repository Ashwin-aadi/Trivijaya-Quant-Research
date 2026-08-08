from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Seasonality in the Indian market involves identifying specific times of the year "
        "when stock performance tends to deviate from long-term trends. This strategy exploits "
        "historical calendar effects by timing investments according to recurring market behaviors."
    )

    def __init__(self, sector_mapping: dict[str, list[str]] = {}, window: int = 365) -> None:
        self._sector_mapping = sector_mapping
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = set(view.symbols).intersection(history["symbol"].to_list())

        sector_returns: dict[str, float] = {}
        for symbol in symbols:
            sector = self._sector_mapping.get(symbol, "Other")
            returns = history.filter(
                pl.col("symbol") == symbol
            ).select(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )["r"].to_list()
            if len(returns) < self._window:
                continue
            sector_returns[sector] = max(returns)

        top_sectors = sorted(sector_returns, key=sector_returns.get, reverse=True)[:5]
        weight = 1.0 / len(top_sectors)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_sectors},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest