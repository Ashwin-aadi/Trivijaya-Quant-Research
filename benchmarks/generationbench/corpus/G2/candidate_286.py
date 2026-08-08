from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Stocks in India may exhibit seasonal trends driven by various economic and social factors. "
        "For example, the monsoon season can impact agriculture-related stocks, while festive seasons might influence retail and consumer goods companies."
    )

    def __init__(self, window: int = 365, lag: int = 1) -> None:
        self._window = window
        self._lag = lag

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_close = {symbol: float(v) for symbol, v in view.latest_close().items()}
        seasonality_signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            symbol_history = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(
                    pl.col("session_date"),
                    (pl.col("close") / pl.col("adj_close").shift(self._lag) - 1).alias("return_ratio"),
                )
            )

            if symbol_history.height < self._window:
                continue

            returns = [float(v[0]) for v in symbol_history["return_ratio"].to_list()]
            max_return = max(returns)
            seasonality_signals[symbol] = max_return

        top_symbols = sorted(seasonality_signals.items(), key=lambda x: x[1], reverse=True)[: self._lag]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest