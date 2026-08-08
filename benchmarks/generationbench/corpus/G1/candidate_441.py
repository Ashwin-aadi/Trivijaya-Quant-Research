from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over long periods. "
        "By tilting our portfolio towards low volatility, we aim to capture this anomaly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        latest_closes = {s: float(v) for s, v in view.latest_close().items()}
        
        returns = (
            history
            .filter(pl.col("symbol").is_in(symbols))
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
        )

        if returns.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = [float(v["volatility"]) for v in returns.to_dicts()]
        symbols_with_vol = list(returns["symbol"].to_list())
        
        sorted_symbols = [s for _, s in sorted(zip(volatility, symbols_with_vol))]
        top_n_lowest_vol = sorted_symbols[:5]

        weights = {s: 1.0 / len(top_n_lowest_vol) for s in top_n_lowest_vol}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest