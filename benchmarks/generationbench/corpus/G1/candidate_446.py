from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySfStrategy(Strategy):
    rationale = (
        "This strategy exploits trends by identifying symbols with high volatility and "
        "following the recent trend. High volatility indicates a strong movement in prices, "
        "which increases the likelihood of a continuation of the trend."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data: dict[str, float] = {}
        for symbol in view.symbols:
            adj_closes = history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].to_list()
            if len(adj_closes) < self._window:
                continue

            volatility = pl.DataFrame({"returns": [(c / adj_closes[i - 1] - 1.0 for i, c in enumerate(adj_closes[1:]))]}
                                      ).select((pl.col("returns").std() * (252 ** 0.5)).alias("volatility"))
            if volatility.height > 0:
                symbol_data[symbol] = float(volatility["volatility"].item())

        sorted_symbols = [s for s, v in sorted(symbol_data.items(), key=lambda item: -item[1])]
        picks = sorted_symbols[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest