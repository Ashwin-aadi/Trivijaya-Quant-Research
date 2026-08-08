from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers over a recent period may continue to "
        "outperform due to momentum effects. This strategy selects the top performers and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window).sort("session_date", descending=True)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        def compute_return_ratio(symbol: str) -> float:
            symbol_history = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), pl.col("adj_close").alias("close")
            )
            latest_close = float(view.latest_close()[symbol])
            if not symbol_history.is_empty():
                returns = (
                    (latest_close - symbol_history["close"].to_list()[0]) / symbol_history["close"].to_list()[0]
                )
            else:
                returns = 0.0
            return returns

        return_ratios = {symbol: compute_return_ratio(symbol) for symbol in view.symbols}
        top_n_symbols = sorted(return_ratios, key=return_ratios.get, reverse=True)[: self._window]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest