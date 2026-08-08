from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which deviate significantly from their historical "
        "average will eventually revert. In a short horizon, this can be exploited by going long "
        "on underperforming stocks and short selling overperforming ones."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history[symbols]
            .group_by("session_date")
            .agg(pl.col("adj_close").mean().alias("mean_close"))
            .sort("session_date", descending=False)
            .select(["symbol", "mean_close"])
        )

        recent_closes = view.closes(lookback=self._window)
        if recent_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_signal: pl.DataFrame = (
            recent_closes
            .select(pl.all().to_list() + ["symbol"])
            .join(mean_close, on="symbol")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_close")).alias("deviation"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)).alias("return"),
            )
            .sort("session_date", descending=False)
        )

        # Calculate mean deviation and std
        mean_deviation = mean_reversion_signal["deviation"].mean().round(4).to_list()[0]
        std_deviation = (
            mean_reversion_signal.select(
                pl.col("deviation").std().alias("std_deviation")
            )
            .collect()
            .height
            > 0
        )
        if not std_deviation:
            return Signal(information_available_at=stamp, weights={})

        # Identify underperforming and overperforming symbols
        underperforming = (
            mean_reversion_signal.filter(pl.col("deviation") < (mean_deviation - 1.5 * std_deviation))
            .select(["symbol"])
            .to_dict(as_pandas=False)["symbol"]
        )
        overperforming = (
            mean_reversion_signal.filter(
                pl.col("deviation") > (mean_deviation + 1.5 * std_deviation)
            )
            .select(["symbol"])
            .to_dict(as_pandas=False)["symbol"]
        )

        # Assign weights
        if underperforming:
            weight = 0.7 / len(underperforming)
            long_weights = {s: weight for s in underperforming}
        else:
            long_weights = {}

        if overperforming:
            short_weight = -0.3 / len(overperforming)
            short_weights = {s: short_weight for s in overperforming}
        else:
            short_weights = {}

        weights = {**long_weights, **short_weights}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest