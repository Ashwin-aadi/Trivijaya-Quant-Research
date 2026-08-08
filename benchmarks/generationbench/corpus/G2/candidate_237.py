from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time. "
        "This is often attributed to risk premium and behavioral biases of investors who may "
        "underprice low-volatility assets."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) == 1:
            return Signal(information_available_at=stamp, weights={})

        mean_closing_prices = closes.select(
            pl.col("session_date").alias("date"),
            pl.mean(pl.all().exclude("date")).suffix("_mean")
        ).collect()

        volatilities = _calculate_volatility(closes)
        sorted_by_volatility = volatilities.sort("volatility", descending=False)

        top_symbols = sorted_by_volatility.head(self._window)["symbol"].to_list()
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(closes: pl.DataFrame) -> pl.DataFrame:
    symbol_to_close_series = {symbol: closes[symbol].to_list() for symbol in view.symbols}
    volatilities = {}

    for symbol, close_prices in symbol_to_close_series.items():
        if len(close_prices) < 2:
            continue
        returns = [float(c / prev_c - 1.0) for c, prev_c in zip(close_prices[1:], close_prices[:-1])]
        volatility = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
        volatilities[symbol] = volatility

    return pl.DataFrame(
        {"symbol": list(volatilities.keys()), "volatility": list(volatilities.values())}
    ).sort("volatility", descending=False)