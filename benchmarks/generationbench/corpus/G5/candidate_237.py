from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy that aims to reduce overall portfolio risk by "
        "favoring assets with lower historical volatility. The rationale behind this approach "
        "is that low-volatility stocks tend to have more stable returns, providing better risk-adjusted performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatility_dict: dict[str, float] = {}
        for symbol in view.symbols:
            close_price_series = (
                history.select(pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_close"))
                .with_columns(
                    (pl.col(f"{symbol}_close") / pl.col(f"{symbol}_close").shift(1) - 1.0)
                    .alias("return")
                )
                .select(pl.col("return"))
            )
            if close_price_series.height < self._window:
                continue
            volatility = (
                (close_price_series["return"].abs().mean() * pl.sqrt(pl.lit(252))).to_list()[0]
            )  # Annualized volatility over the window period
            volatility_dict[symbol] = volatility

        if not volatility_dict:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [
            symbol for symbol, _ in sorted(volatility_dict.items(), key=lambda item: item[1])
        ]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest