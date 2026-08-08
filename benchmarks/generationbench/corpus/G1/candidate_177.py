from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over long periods. By tilting "
        "the portfolio towards low volatility, we aim to capture these outperformance effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatility: dict[str, float] = {}
        for symbol in view.symbols:
            daily_returns = (
                history.lazy()
                .filter(pl.col("symbol") == symbol)
                .select(
                    (pl.col("adj_close").shift(-1) / pl.col("adj_close")) - 1.0
                )
                .collect()["arr_0"]
                .to_list()
            )
            if len(daily_returns) < self._window:
                continue

            vol = ((sum([x**2 for x in daily_returns]) / self._window) ** 0.5)
            volatility[symbol] = float(vol)

        sorted_symbols = [k for k, v in sorted(
            volatility.items(), key=lambda item: item[1]
        )][:3]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

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