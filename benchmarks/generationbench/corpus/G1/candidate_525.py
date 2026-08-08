from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often undervalued and exhibit less price fluctuation. "
        "By tilting towards low volatility, we aim to capture the risk premium associated "
        "with these assets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        volatilities = []
        for symbol in symbols:
            prices = history.filter(pl.col("symbol") == symbol)[["session_date", "adj_close"]]
            returns = (prices.with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))).sort(
                "session_date"
            ).drop_nulls().select(pl.col("r").to_list())
            if len(returns) < self._window:
                continue
            volatility = (pl.Series(returns).std()).round(4)
            volatilities.append((symbol, float(volatility)))

        sorted_volatilities = sorted(volatilities, key=lambda x: x[1])
        picks = [s for s, v in sorted_volatilities[:5]]
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