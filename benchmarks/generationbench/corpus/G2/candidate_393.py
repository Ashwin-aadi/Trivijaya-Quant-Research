from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is often attributed to risk-return trade-offs and market sentiment towards lower-risk assets."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(closes) < self._window:
                continue

            # Calculate daily returns
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]

            # Standard deviation of returns as a proxy for volatility
            vol = pl.DataFrame({"returns": returns}).select(
                (pl.col("returns").std().alias("volatility"))
            ).collect().item()
            volatilities[symbol] = vol

        sorted_symbols = sorted(volatilities, key=volatilities.get)
        top_n_low_vol_symbols = sorted_symbols[:5]

        if not top_n_low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_low_vol_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest