from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and can provide more stable returns "
        "over time. This strategy aims to tilt the portfolio towards these low-risk stocks."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        volatilities = [
            float(
                (history[symbol].select((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("ret"))
                 .select(pl.col("ret").std()).collect().row(0)[0])
            )
            for symbol in symbols
        ]

        sorted_symbols = [symbol for _, symbol in sorted(zip(volatilities, symbols))]
        top_low_volatility = sorted_symbols[:5]
        if not top_low_volatility:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_low_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_low_volatility},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest