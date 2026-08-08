from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when asset prices return to their mean after a period of "
        "deviation. In the Indian market, this can be observed in daily price movements where "
        "extreme highs or lows revert towards historical means over time."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 3.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in df.select("adj_close").to_series().drop_nulls()]
            z_score = (prices[-1] - pl.col("adj_close").mean()) / pl.col(
                "adj_close"
            ).std()
            if abs(z_score) > self._z_score_threshold:
                symbol_data[symbol] = {
                    "last_price": prices[-1],
                    "mean": pl.col("adj_close").mean().item(),
                    "std_dev": pl.col("adj_close").std().item(),
                    "z_score": z_score,
                }

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        target_symbols = sorted(symbol_data.keys(), key=lambda x: abs(symbol_data[x]["z_score"]))[
                         :5
                       ]

        weight = 1.0 / len(target_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in target_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest