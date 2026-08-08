from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where volatility scales with market trends. "
        "By scaling position sizes based on current implied or historical volatility, it aims "
        "to balance risk and reward during trending periods."
    )

    def __init__(self, trend_window: int = 200, vol_window: int = 20) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window + self._vol_window)

        if closes.height < self._trend_window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple moving averages
        sma_short = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias(f"sma_{self._trend_window}"))
            )
            .with_columns(
                (
                    pl.col(f"sma_{self._trend_window}")
                    / pl.col(f"sma_{self._trend_window}").shift(self._trend_window)
                    - 1.0
                ).alias("sma_diff")
            )
        ).collect()

        # Identify trending symbols based on moving average crossover
        buys = sma_short.filter(
            (pl.col("sma_diff") > 0) & (pl.col(f"sma_{self._trend_window}") < pl.col(f"sma_{self._trend_window}").shift(1))
        )["symbol"].to_list()

        sells = sma_short.filter(
            (pl.col("sma_diff") < 0) & (pl.col(f"sma_{self._trend_window}") > pl.col(f"sma_{self._trend_window}").shift(1))
        )["symbol"].to_list()

        # Calculate historical volatility
        log_returns = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return")
            )
        ).collect()

        vol = log_returns.group_by("symbol").agg(pl.col("log_return").std().alias(f"vol_{self._vol_window}")).to_pandas()

        # Scale positions based on volatility
        weight_dict = {}
        for symbol in buys + sells:
            if symbol not in vol.columns:
                continue
            vol_symbol = float(vol[vol["symbol"] == symbol]["vol_" + str(self._vol_window)].iloc[-1])
            position_size = 1.0 / (vol_symbol + 1)
            weight_dict[symbol] = position_size

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weight_dict.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest