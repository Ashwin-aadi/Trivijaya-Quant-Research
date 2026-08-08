from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the tendency of financial markets to exhibit mean-reverting "
        "behavior within trends. By scaling trend following based on volatility, it aims to "
        "balance risk and potential returns."
    )

    def __init__(self, lookback: int = 20, threshold_days: int = 180) -> None:
        self._lookback = lookback
        self._threshold_days = threshold_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback + self._threshold_days)
        if history.is_empty() or history.height < self._lookback + self._threshold_days:
            return Signal(information_available_at=stamp, weights={})

        volatility_20d = (
            history.with_columns(
                (pl.col("close").shift(-1) - pl.col("open")).abs().alias("price_diff")
            )
            .with_column((pl.col("price_diff") / pl.col("adj_close").shift(1)).alias("return"))
            .with_columns(
                (pl.col("return").rolling_std(window_size=self._lookback).alias(f"vol_20d"))
            )
        )

        vol_ma = (
            history.with_column((pl.col("close") - pl.col("open")).abs().mean().over(pl.date_range)).alias(f"vol_ma_{self._threshold_days}")
        )

        positions = {}
        for symbol in view.symbols:
            if symbol not in volatility_20d.columns or symbol not in vol_ma.columns:
                continue
            vol_20d_values = [float(v) for v in volatility_20d[symbol].to_list()[-self._lookback:]]
            vol_ma_value = float(vol_ma[symbol].tail(1))
            if len(vol_20d_values) < self._lookback:
                continue
            position_size = 1.0 / (1 + abs(self._threshold_days - view.as_of.year)) if vol_20d_values[-1] > vol_ma_value else 1.0 / len(view.symbols)
            positions[symbol] = position_size

        return Signal(information_available_at=stamp, weights=positions)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest