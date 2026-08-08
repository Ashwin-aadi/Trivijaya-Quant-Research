from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Stock markets often exhibit seasonal patterns driven by calendar effects such as "
        "quarterly earnings reports, end-of-year tax considerations, and holiday-related trading "
        "behavior. The NIFTY 100 indices may show higher returns in certain months of the year due to "
        "these factors."
    )

    def __init__(self, seasonality_window: int = 365, top_n_symbols: int = 5) -> None:
        self._seasonality_window = seasonality_window
        self._top_n_symbols = top_n_symbols

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonality_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = {}
        for symbol in view.symbols:
            symbol_df = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            symbol_returns[symbol] = (
                (symbol_df["close"].tail(1) / symbol_df["close"].head(1)) - 1.0
            ).mean()

        sorted_symbols = [
            s for _, s in sorted(symbol_returns.items(), key=lambda item: item[1], reverse=True)
        ]
        top_symbols = sorted_symbols[: self._top_n_symbols]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest