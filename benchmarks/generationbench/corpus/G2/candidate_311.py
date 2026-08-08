from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumAndVolatility(Strategy):
    rationale = (
        "This strategy leverages the idea that stocks with high momentum (recent price increase) "
        "and low volatility tend to continue their current trend. High momentum suggests strong "
        "buying pressure and low volatility indicates stable prices, both of which are often positive "
        "signals for future returns."
    )

    def __init__(self, momentum_window: int = 20, vol_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)

        if history.is_empty() or history.height < self._momentum_window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["close"]
        returns = (closes / closes.shift(1) - 1.0).drop_nulls().to_list()[self._volatility_window:]
        
        volatilities = [pl.col("adj_close").std().alias("volatility")] * len(view.symbols)
        vol_history = history.select(*volatilities)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in vol_history.columns:
                continue
            momentum_score = max(returns) if returns else 0.0
            volatility_score = float(vol_history[symbol].select("volatility").item())
            combined_score = (momentum_score - min(momentum_score, 0)) / (1 + volatility_score)
            if combined_score > 0:
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest