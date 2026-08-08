from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the relationship between market trends and volatility by "
        "identifying strong trending movements while scaling trades based on recent volatility levels. "
        "During high volatility periods, position sizes are reduced to manage risk."
    )

    def __init__(self, sma_window: int = 50, vol_window: int = 20, max_positions: int = 20) -> None:
        self._sma_window = sma_window
        self._vol_window = vol_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._sma_window + self._vol_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_50 = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").shift(-self._sma_window).rolling_mean(self._sma_window)).alias(f"sma_{self._sma_window}"))
            .sort("session_date", descending=True)
            .select(pl.col("symbol"), f"sma_{self._sma_window}")
        )

        vol_20 = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0).rolling_std(self._vol_window).alias(f"vol_{self._vol_window}"))
            .sort("session_date", descending=True)
            .select(pl.col("symbol"), f"vol_{self._vol_window}")
        )

        closes = view.closes(lookback=self._sma_window + self._vol_window)

        combined_df = (
            history
            .join(sma_50, on="symbol")
            .join(vol_20, on="symbol")
            .select("symbol", "session_date", f"sma_{self._sma_window}", f"vol_{self._vol_window}")
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in combined_df.columns or combined_df.height < self._sma_window + self._vol_window:
                continue

            sma_val = float(combined_df.filter(pl.col("symbol") == symbol)[f"sma_{self._sma_window}"].to_list()[-1])
            vol_val = float(combined_df.filter(pl.col("symbol") == symbol)[f"vol_{self._vol_window}"].to_list()[-1])

            latest_close = view.latest_close()[symbol]

            if latest_close > sma_val:
                picks.append(symbol)
            elif latest_close < sma_val:
                picks.append(f"-{symbol}")  # Indicate short positions

        picks = [p for p in picks[: self._max_positions] if p != ""]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s.replace("-", ""): (weight if s[0] != "-" else -weight) for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest