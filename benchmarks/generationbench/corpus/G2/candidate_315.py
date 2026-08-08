from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "Highly liquid stocks are expected to be more efficiently priced and less volatile. "
        "By equal-weighting the most liquid stocks, we aim to capture this efficiency without "
        "overweighting risk-prone, less liquid securities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate volume-weighted average price for each symbol
        vwaps = (
            history.select(["symbol", "session_date", "adj_close", "volume"])
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") * pl.col("volume").sum() / pl.col("volume")).alias("vwap"),
            )
        )

        # Calculate total volume for all symbols
        total_volume = history.select(["session_date", "volume"]).group_by("session_date").sum()

        # Compute liquidity score as the ratio of VWAP to average daily trading volume
        vwaps_with_scores = (
            vwaps.join(total_volume, on="session_date")
            .with_columns(
                (pl.col("vwap") / pl.col("volume")).alias("liquidity_score"),
            )
        )

        # Filter out symbols with insufficient history or low liquidity scores
        valid_symbols = [
            symbol for symbol in view.symbols if symbol in vwaps_with_scores["symbol"].to_list()
        ]
        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal-weight these symbols
        weight_per_symbol = 1.0 / len(valid_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol for symbol in valid_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest