from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumDecile(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting the top deciles of stocks "
        "based on their recent price performance over a 6-12 month period. Stocks with strong "
        "recent returns are more likely to continue outperforming, as observed in equity markets."
    )

    def __init__(self, window: int = 180, top_decile: float = 0.1) -> None:
        self._window = window
        self._top_decile = top_decile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        cumulative_returns = (
            (closes / closes.shift(self._window) - 1.0).drop_nulls().to_list()
        )
        symbols = view.symbols

        if len(cumulative_returns) < self._window or len(symbols) != len(
            cumulative_returns
        ):
            return Signal(information_available_at=stamp, weights={})

        ranked_symbols = [
            (symbol, cum_ret)
            for symbol, cum_ret in zip(symbols, cumulative_returns)
            if not pl.col("cum_ret").is_nan()
        ]
        ranked_symbols.sort(key=lambda x: x[1], reverse=True)

        top_n_symbols = int(len(symbols) * self._top_decile)
        top_performers = [symbol for symbol, _ in ranked_symbols[:top_n_symbols]]

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest