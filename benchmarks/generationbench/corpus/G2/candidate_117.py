from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is based on the idea that low volatility often indicates lower risk and potentially "
        "lower return requirements from investors, leading to higher valuations and thus better returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            vol = (
                history[symbol]
                .select(
                    (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().mean()
                )
                .item()
            )
            low_vol_symbols.append((symbol, vol))

        # Filter out the symbols with very high volatility
        threshold = max(v[1] for v in low_vol_symbols) * 0.95
        low_vol_symbols = [s for s in low_vol_symbols if s[1] <= threshold]

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = sorted(low_vol_symbols, key=lambda x: x[1])
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest