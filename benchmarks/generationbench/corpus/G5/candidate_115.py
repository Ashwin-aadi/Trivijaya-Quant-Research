from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the stock is experiencing reduced volatility "
        "relative to its recent price range. This can be an opportunity for mean reversion, "
        "as prices often revert back to their mean after a period of high compressions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        ranges = (
            history.group_by("symbol")
            .agg(
                (pl.col("high") - pl.col("low")).alias("range"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().alias("price_change"),
            )
            .sort("symbol")
        )

        symbols = ranges["symbol"].to_list()
        range_values = [float(v[0]) for v in ranges.select("range").rows()]
        price_change_values = [float(v[0]) if v[0] != 0 else 1e-6 for v in ranges.select("price_change").rows()]

        compressed_symbols: list[str] = []
        for i, (symbol, range_val, change) in enumerate(zip(symbols, range_values, price_change_values)):
            if range_val / change < 2.0:
                compressed_symbols.append(symbol)

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest