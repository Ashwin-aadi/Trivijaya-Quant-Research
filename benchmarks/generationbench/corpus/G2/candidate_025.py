from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when a stock's price tends to return to its historical average. "
        "By identifying stocks that are trading far from their 20-day mean, we can exploit this "
        "tendency for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").mean()).alias("avg_close"))
        )
        recent_closes = view.closes()

        symbol_avg_close_dict: dict[str, float] = {
            row["symbol"]: float(row["avg_close"])
            for row in avg_close.to_dicts()
        }

        score = {}
        for symbol in view.symbols:
            if symbol not in recent_closes.columns or symbol not in symbol_avg_close_dict:
                continue
            recent_closings = [float(v) for v in recent_closes[symbol].drop_nulls().to_list()]
            if len(recent_closings) < self._window:
                continue

            avg_close_value = symbol_avg_close_dict[symbol]
            latest_close_value = float(recent_closings[-1])
            deviation_from_mean = abs(latest_close_value - avg_close_value)
            score[symbol] = deviation_from_mean

        symbols_to_trade = [
            symbol
            for symbol, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols_to_trade}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest