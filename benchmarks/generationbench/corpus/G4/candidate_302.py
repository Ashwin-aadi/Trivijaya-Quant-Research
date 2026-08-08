from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits trends in the Indian market by scaling trade sizes according "
        "to realized volatility. High volatility periods are associated with more pronounced and "
        "persistent price movements, making it favorable to enter larger positions in stocks "
        "with strong trends."
    )

    def __init__(self, trend_window: int = 50, vol_lookback: int = 20) -> None:
        self._trend_window = trend_window
        self._vol_lookback = vol_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._trend_window + self._vol_lookback)
        if history.height < self._trend_window + self._vol_lookback:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]

        realized_volatility: dict[str, float] = {}
        for symbol in symbols:
            prices = history.filter(pl.col("symbol") == symbol)[
                ["session_date", "adj_close"]
            ]
            returns = (prices["adj_close"] / prices["adj_close"].shift(1) - 1.0).drop_nulls()
            if returns.height < self._vol_lookback:
                continue
            vol = returns.std().round(4)
            realized_volatility[symbol] = float(vol)

        trend_strength: dict[str, float] = {}
        for symbol in symbols:
            prices = history.filter(pl.col("symbol") == symbol)[["session_date", "adj_close"]]
            prices = (
                prices.sort(by="session_date")
                .with_columns((pl.col("adj_close").shift(-1) - pl.col("adj_close")).alias("return"))
                .sort(by=["session_date"])
                .tail(self._trend_window)
            )
            returns = [float(r) for r in prices["return"].to_list()]
            if len(returns) < self._trend_window:
                continue
            tsi = sum(returns[-25:]) / 100.0 * 1000.0
            trend_strength[symbol] = float(tsi)

        ranked_symbols = [
            (symbol, realized_volatility[symbol] * trend_strength[symbol])
            for symbol in symbols
            if symbol in realized_volatility and symbol in trend_strength
        ]
        ranked_symbols.sort(key=lambda x: x[1], reverse=True)
        top_n = 30

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols[:top_n])
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for _, (symbol, _) in zip(range(top_n), ranked_symbols)
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest