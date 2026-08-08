from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength against the market are more likely to continue "
        "outperforming. This is based on the assumption that strong stocks will tend to remain "
        "strong in a trending market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative strength for each stock
        def calculate_strength(symbol: str) -> float:
            closes_history = view.history().filter(
                (pl.col("symbol") == symbol) & (pl.col("session_date").is_in(closes.columns))
            )
            if closes_history.height < self._window:
                return 0.0

            mean_close = closes_history.select(pl.col("adj_close").mean()).item()
            strength = (
                max(
                    [
                        float(c)
                        for c in (closes[symbol].drop_nulls().to_list()[-self._window:])
                    ]
                )
                / mean_close
            ) - 1.0

            return strength if strength > 0 else 0.0

        strengths = {symbol: calculate_strength(symbol) for symbol in view.symbols}
        sorted_stocks = sorted(strengths.items(), key=lambda x: x[1], reverse=True)

        # Select the top stocks
        top_stocks = [stock for stock, _ in sorted_stocks[:5]]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest