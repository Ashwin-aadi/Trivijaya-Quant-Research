from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which are far from their recent means "
        "are likely to move back towards them. This strategy identifies stocks whose prices "
        "have deviated significantly from their trailing average and bets on a reversion."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            data = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
                )
                .sort("session_date")
                .with_columns((pl.col("r").rolling_mean(window_size=self._window)).alias(f"mean_r"))
            )

            latest_price = view.latest_close()[symbol]
            if data.shape[0] < self._window:
                continue

            recent_return = float(data.select(pl.last("r")).item())
            mean_return = float(data.select(pl.last("mean_r")).item())

            deviation = abs(recent_return - mean_return)
            symbol_prices[symbol] = (recent_return, mean_return, deviation)

        top_symbols = sorted(symbol_prices.items(), key=lambda x: x[1][2], reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        signal_weights = {s: weight for s, _ in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest