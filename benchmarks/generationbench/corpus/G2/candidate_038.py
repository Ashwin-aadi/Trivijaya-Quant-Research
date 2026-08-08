from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the tendency for asset prices to revert to "
        "their mean after a period of high volatility. High volatility periods often indicate "
        "greater uncertainty and potential for large price movements in either direction, which can be capitalized upon."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history
            .with_column((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
            .sort("session_date")
        )

        symbols_with_sufficient_data = [symbol for symbol in view.symbols if all(history[symbol].is_not_null().to_list())]

        # Calculate mean absolute returns
        mean_abs_returns = history.select(
            pl.col(symbols_with_sufficient_data).mean().alias("mean_abs_return")
        ).collect()["mean_abs_return"].to_list()[0]

        # Identify symbols with above-threshold returns
        signals: dict[str, float] = {}
        for symbol in symbols_with_sufficient_data:
            if history[symbol].mean().abs() > mean_abs_returns * 1.5:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        normalized_weights = {symbol: weight / total_weight for symbol, weight in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in normalized_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest