from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySMTF(Strategy):
    rationale = (
        "This strategy identifies trending stocks by combining moving average crossovers with volatility scaling. "
        "It aims to capitalize on significant price movements while adjusting positions based on the volatility of each stock."
    )

    def __init__(self, sma_window: int = 50, vol_lookback: int = 20) -> None:
        self._sma_window = sma_window
        self._vol_lookback = vol_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._vol_lookback + max(self._sma_window, 1))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history.select(["session_date", "symbol", "close"])
        sma_50 = (
            filtered_history.groupby("symbol")
            .agg(
                (pl.col("close").shift(-self._sma_window).mean().alias("sma"))
            )
            .sort("sma", descending=True)
            .select("symbol")
            .to_series()
            .to_list()[:3]
        )

        volatilities = (
            filtered_history.groupby("symbol")
            .agg(
                (pl.col("close").std().alias("volatility"))
            )
            .select("symbol", "volatility")
            .with_columns(pl.col("volatility") / pl.col("volatility").mean() < 1.0)
        )

        top_symbols = [s for s in sma_50 if s in volatilities.columns and float(volatilities[volatilities["symbol"] == s]["volatility"].to_list()[0]) <= 1.0]
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest