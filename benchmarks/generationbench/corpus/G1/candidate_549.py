from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often under-owned and can provide downside protection. "
        "By tilting towards low-volatility stocks, we aim to enhance portfolio stability."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            returns = (history[f"{symbol}_close"] / history[f"{symbol}_close"].shift(1) - 1.0).to_list()
            if len(returns) < self._window:
                continue
            volatility = pl.DataFrame({"returns": returns}).select(
                ((pl.col("returns").std() * (252**0.5)).alias("volatility"))
            ).item()
            volatilities[symbol] = float(volatility)

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [k for k, _ in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest