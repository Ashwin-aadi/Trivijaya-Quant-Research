from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because low-volatility stocks are often considered less risky and may benefit from "
        "risk premiums or better risk-adjusted returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            close_series = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(close_series) < self._window:
                continue

            volatility = pl.DataFrame({"returns": [(close / close.shift(1) - 1.0).drop_nulls() for close in close_series]})
            avg_volatility = volatility["returns"].std().round(4)
            low_vol_symbols.append((symbol, avg_volatility))

        sorted_symbols = sorted(low_vol_symbols, key=lambda x: x[1])
        top_n_symbols = [symbol for symbol, _ in sorted_symbols[:5]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest