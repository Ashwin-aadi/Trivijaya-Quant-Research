from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Equities with lower historical volatility tend to outperform those with higher "
        "volatility over the long term. By tilting our portfolio towards low-volatility "
        "stocks, we aim to enhance overall performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volatility = ((pl.Series(prices).rolling_std(window=self._window))[-1]).item()
            symbol_volatility[symbol] = volatility

        symbols_sorted_by_volatility = sorted(symbol_volatility.items(), key=lambda x: x[1])
        picks = [symbol for symbol, _ in symbols_sorted_by_volatility[:5]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest