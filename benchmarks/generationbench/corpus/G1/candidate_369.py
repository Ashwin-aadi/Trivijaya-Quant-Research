from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting aims to capitalize on the empirical observation that "
        "low-volatility stocks tend to outperform high-volatility stocks over time."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            daily_returns = (
                history.select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .filter(pl.col("symbol") == symbol)
                .select(pl.col("return"))
            )
            if daily_returns.height < self._window:
                continue
            mean_return = float(daily_returns.mean().item())
            volatility = float(daily_returns.std().item())
            low_vol_symbols.append((symbol, volatility))

        sorted_symbols = sorted(low_vol_symbols, key=lambda x: x[1])
        top_n_low_vol_symbols = [s for s, v in sorted_symbols[:5]]
        if not top_n_low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_low_vol_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest