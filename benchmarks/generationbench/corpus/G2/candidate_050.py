from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the tendency for stocks in an uptrend to "
        "continue trending up and those in a downtrend to continue down. By scaling trends by "
        "volatility, we can enter when the underlying security is likely to continue its direction."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["session_date"]) < 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Compute rolling mean and std of returns
        rolling_mean = (
            history.with_columns(
                (pl.col("return").rolling_mean(self._window)).alias(f"mean_{self._window}")
            )
            .with_columns(
                (pl.col("return").rolling_std(self._window)).alias(f"std_{self._window}")
            )
        )

        # Compute z-score
        rolling_z_score = (
            rolling_mean.with_columns(
                (pl.col("return") - pl.col(f"mean_{self._window}") / pl.col(f"std_{self._window}")).alias(f"z_score")
            )
        )

        # Identify trending symbols with high z-score
        picks: list[str] = []
        for symbol in view.symbols:
            if f"{symbol}_z_score" not in rolling_z_score.columns:
                continue
            z_scores = [float(v) for v in rolling_z_score[f"{symbol}_z_score"].to_list()]
            last_z_score = max(z_scores, default=0)
            if last_z_score > self._z_score_threshold:
                picks.append(symbol)

        picks = list(set(picks))  # Remove duplicates
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