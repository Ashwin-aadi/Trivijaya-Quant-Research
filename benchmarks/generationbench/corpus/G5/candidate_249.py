from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "By tilting the portfolio towards low volatility, we can potentially enhance returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        volatility: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.select(pl.col("symbol") == symbol).select("adj_close")
            if symbol_history.height < self._window:
                continue
            close_values = [float(v) for v in symbol_history.drop_nulls().to_list()]
            volatility[symbol] = ((pl.Series(close_values).std()) / pl.Series(close_values).mean()).round(4)

        sorted_symbols = [k for k, _ in sorted(volatility.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest