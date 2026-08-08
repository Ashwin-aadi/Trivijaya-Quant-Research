from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Stocks often exhibit seasonality based on calendar effects. For example, certain "
        "industries may see increased trading volumes or prices at specific times of the year."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        mean_prices = (
            history.groupby("symbol")
            .agg(pl.col("adj_close").mean().alias("mean_price"))
            .to_dict(False)
        )

        seasonality_factors = {}
        for symbol in symbols:
            if symbol not in mean_prices:
                continue
            prices = view.closes(lookback=self._window).select(symbol)
            price_series = [float(v) for v in prices.to_dict(True)[symbol]]
            if len(price_series) < self._window:
                continue

            mean_price = mean_prices[symbol]["mean_price"]
            seasonality_factor = (
                (price_series[-10:] - mean_price).sum() / (len(price_series) * mean_price)
            )
            seasonality_factors[symbol] = seasonality_factor

        if not seasonality_factors:
            return Signal(information_available_at=stamp, weights={})

        threshold = 0.5
        weight_per_symbol = 1.0 / len(seasonality_factors)

        selected_symbols = [
            symbol for symbol, factor in seasonality_factors.items() if abs(factor) > threshold
        ]

        weights = {symbol: weight_per_symbol for symbol in selected_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().get("session_date", date(1900, 1, 1))
    assert isinstance(newest, date)
    return newest