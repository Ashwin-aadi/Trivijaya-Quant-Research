from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion is a fundamental trading principle that suggests prices "
        "tend to move towards their long-term average after extreme deviations. "
        "By identifying symbols that have moved far from their trailing mean, "
        "we can capitalize on the tendency for prices to revert."
    )

    def __init__(self, window: int = 50, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing mean and standard deviation for each symbol
        means = (
            history.groupby("symbol")
            .agg(
                pl.col("adj_close").mean().alias("trailing_mean"),
                pl.col("adj_close").std().alias("trailing_std"),
            )
            .to_pandas()
        )

        # Calculate the z-score to identify extreme deviations
        def calculate_z_score(row):
            adj_close = float(row["adj_close"])
            trailing_mean = row["trailing_mean"]
            trailing_std = row["trailing_std"]
            if not pd.isnull(trailing_mean) and not pd.isnull(trailing_std):
                return (adj_close - trailing_mean) / trailing_std
            return None

        history = (
            history.with_columns(
                pl.col("adj_close").apply(calculate_z_score).alias("z_score")
            )
            .to_pandas()
        )

        # Filter symbols with extreme z-scores indicating potential reversion
        recent_closes = view.closes(lookback=None)
        picks: list[str] = []
        for symbol in means["symbol"]:
            if not recent_closes.get_column(symbol).is_null().any():
                row = means[means["symbol"] == symbol].iloc[0]
                z_score = float(row["z_score"])
                if abs(z_score) > self._threshold:
                    picks.append(symbol)

        # Ensure the picked symbols are in the current history
        valid_picks = [p for p in picks if p in recent_closes.columns]
        if not valid_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(valid_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in valid_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest