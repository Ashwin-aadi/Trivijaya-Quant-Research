from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High historical volatility can indicate a market that is in the early stages of a trend. "
        "By scaling our position according to the historical volatility, we aim to capture gains from established trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            close_series = history.select(pl.col("symbol") == symbol).select(
                "adj_close"
            )
            price_changes = (
                close_series["adj_close"].diff().drop_nulls().to_list()[1:]
            )
            volatility = pl.Series(price_changes).std()
            if not pl.is_nan(volatility):
                symbol_data[symbol] = (volatility, history["close"][0])

        sorted_symbols = [
            k for k, v in sorted(symbol_data.items(), key=lambda item: -item[1][0])
        ][:2]
        weights = {symbol: 1.0 / len(sorted_symbols) for symbol in sorted_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest