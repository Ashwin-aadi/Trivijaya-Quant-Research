from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Markets exhibit stronger trends during periods of higher volatility. "
        "This strategy leverages this phenomenon by scaling trades based on recent volatility metrics. "
        "During high-volatility periods, it increases exposure to trending assets; "
        "during low-volatility periods, it reduces or exits positions."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg((pl.col("daily_return").std().alias("volatility"),))
        )

        # Get latest close prices
        closes = view.closes()

        # Rank symbols based on recent trend strength relative to volatility
        rank_map: dict[str, float] = {}
        for symbol in history.symbol.unique():
            if symbol not in closes.columns:
                continue
            last_close = float(closes[symbol][0])
            vol = history[history.symbol == symbol]["volatility"].item()
            daily_return = (last_close - history[history.symbol == symbol].select("adj_close").to_series().mean()) / vol
            rank_map[symbol] = daily_return

        ranked_symbols = sorted(rank_map.items(), key=lambda x: x[1], reverse=True)[: self._top_n]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in ranked_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest