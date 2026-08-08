from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedAmbitiousDesign(Strategy):
    rationale = (
        "This strategy integrates momentum, earnings growth potential, volume-weighted average price (VWAP), and volatility to create a balanced portfolio. It aims to leverage short-term opportunities while managing risk through multiple exit rules."
    )

    def __init__(self, window: int = 20, top_n_percent: float = 0.1) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate momentum
        returns_20d = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        momentum = (
            history.select(
                pl.col("symbol"), returns_20d, pl.col("adj_close").mean().over("symbol").alias("VWAP")
            ).sort("session_date", descending=True)
            .group_by("symbol")
            .agg([pl.col("r").quantile(0.5).alias("M"), "VWAP"])
            .with_columns(
                (pl.col("r") - pl.col("r").mean().over("symbol")).abs().alias("abs_r"),
                ((pl.col("adj_close") / pl.col("adj_close").rolling_mean(window_size=self._window, center=False) - 1.0).std().over("symbol")) * 2.0
            )
        )

        # Calculate earnings growth potential (simulated for demonstration)
        earnings_growth = (
            history.select(pl.col("symbol"), "adj_close")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0).alias("EGR")
            )
        )

        # Calculate volatility
        volatilities = (
            history.select(pl.col("symbol"), "adj_close", "volume")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std().over("symbol") * 2.0).alias("VOL")
            )
        )

        # Combine metrics
        composite_score = (
            momentum.join(earnings_growth, on="symbol")
            .join(volatilities, on="symbol")
            .with_columns(
                (pl.col("M") * 0.4 + pl.col("EGR") * 0.25 + pl.col("VWAP") * 0.2 - pl.col("VOL") * 0.1).alias("composite_score"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0) > 0).cast(pl.int32).alias("momentum_positive")
            )
        )

        # Select top N% with positive momentum
        composite_score = (
            composite_score.filter(
                (pl.col("momentum_positive") == 1)
            ).sort(
                "composite_score", descending=True
            )
        ).select(
            pl.col("symbol"), "composite_score"
        )

        top_n_symbols = int(composite_score.height * self._top_n_percent)
        top_symbols = [row["symbol"] for row in composite_score.head(top_n_symbols).to_dicts()]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest