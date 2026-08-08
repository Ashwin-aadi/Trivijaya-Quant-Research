from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrading(Strategy):
    rationale = (
        "Stock prices may exhibit seasonal patterns due to recurring events such as "
        "holiday seasons or earnings reports. If the market tends to be more active and "
        "prices higher during certain months of the year, a strategy that buys into these "
        "seasons could potentially generate returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            factor = _calculate_seasonal_factor(history[history["symbol"] == symbol])
            if factor is None:
                continue
            seasonal_factors[symbol] = factor

        sorted_factors = sorted(seasonal_factors.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_factors[:5]]
        
        weight = 1.0 / len(top_symbols) if top_symbols else 0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_seasonal_factor(history: pl.DataFrame) -> float | None:
    close_series = history.select(pl.col("adj_close"))
    if close_series.height < 20:
        return None

    avg_close = close_series.mean().item()
    jan_avg = close_series.filter((pl.col("session_date").dt.month() == 1)).mean().item()
    octo_avg = close_series.filter((pl.col("session_date").dt.month() == 10)).mean().item()

    if abs(jan_avg - avg_close) > 5 or abs(octo_avg - avg_close) > 5:
        return None

    seasonal_factor_jan = (jan_avg / avg_close - 1.0).round(4)
    seasonal_factor_octo = (octo_avg / avg_close - 1.0).round(4)

    if abs(seasonal_factor_jan) < 0.1 and abs(seasonal_factor_octo) < 0.1:
        return None

    return max(seasonal_factor_jan, seasonal_factor_octo)