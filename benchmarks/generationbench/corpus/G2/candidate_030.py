from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength against the market index are expected to outperform "
        "over the short term. This is based on the notion that strong stocks tend to continue "
        "outperforming during bull markets and underperforming during bear markets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty100_history = history.select(pl.col("symbol").is_in(view.symbols))
        nifty100_closes = nifty100_history.group_by("symbol").agg(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
        )
        market_avg_return = (
            nifty100_closes["daily_return"].mean().item()
        )  # Get the average return
        symbols_with_data = set(nifty100_history.columns[2:]) & set(view.symbols)

        if not symbols_with_data:
            return Signal(information_available_at=stamp, weights={})

        filtered_history = nifty100_history.select(
            pl.col("symbol").is_in(symbols_with_data)
        )

        relative_strengths: list[tuple[str, float]] = []
        for symbol in symbols_with_data:
            symbol_closes = filtered_history.filter(pl.col("symbol") == symbol).select(
                "daily_return"
            ).to_series()
            if len(symbol_closes) < self._window:
                continue
            avg_daily_return = symbol_closes.mean().item()
            relative_strength = (avg_daily_return - market_avg_return) / market_avg_return
            relative_strengths.append((symbol, relative_strength))

        sorted_strongest = sorted(relative_strengths, key=lambda x: x[1], reverse=True)
        strongest_symbols = [symbol for symbol, _ in sorted_strongest[:5]]

        if not strongest_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strongest_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strongest_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest