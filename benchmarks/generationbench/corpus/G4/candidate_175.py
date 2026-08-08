from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "This strategy exploits the market behavior of dispersion or range compression by "
        "identifying sectors or stocks where price volatility is increasing (dispersion) or "
        "decreasing (range compression). It leverages technical indicators like Bollinger Bands "
        "to confirm these conditions and enter long positions in dispersed sectors and short "
        "positions in compressed ranges."
    )

    def __init__(self, window: int = 20, threshold_dispersion: float = 0.2, threshold_compression: float = 0.3) -> None:
        self._window = window
        self._threshold_dispersion = threshold_dispersion
        self._threshold_compression = threshold_compression

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate Bollinger Bands
        bbands_df = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(window_size=self._window)).alias(f"sma_{self._window}"),
                (pl.col("adj_close").rolling_std(window_size=self._window)).alias(f"std_{self._window}")
            )
            .with_columns([
                ((pl.col(f"sma_{self._window}") + 2 * pl.col(f"std_{self._window}")).alias("upper_bb")),
                ((pl.col(f"sma_{self._window}") - 2 * pl.col(f"std_{self._window}")).alias("lower_bb"))
            ])
        )

        # Determine Range Compression
        bbands_df = (
            bbands_df.with_columns(
                (pl.col("adj_close") / pl.col("sma_" + str(self._window)) - 1).alias("z_score"),
                ((pl.col("high") - pl.col("low")) / (2 * pl.col(f"std_{self._window}")).round(4)).alias("price_range_ratio")
            )
        )

        # Rank stocks based on compression
        compressed_ranks = bbands_df.filter(
            pl.col("price_range_ratio").le(self._threshold_compression)
        ).sort("session_date", descending=True).select(
            "symbol",
            (pl.col("price_range_ratio") / self._threshold_compression * 100).alias("compression_rank")
        )

        # Rank stocks based on dispersion
        dispersion_ranks = bbands_df.filter(
            pl.col(f"std_{self._window}").change().gt(self._threshold_dispersion)
        ).sort("session_date", descending=True).select(
            "symbol",
            (pl.col(f"std_{self._window}") / self._threshold_dispersion * 100).alias("dispersion_rank")
        )

        # Combine and select top N candidates
        combined_ranks = compressed_ranks.hstack(dispersion_ranks)
        selected_compressed = [row["symbol"] for row in combined_ranks.sort("compression_rank", descending=True).head(self._top_n)["symbol"]]
        selected_dispersed = [row["symbol"] for row in combined_ranks.sort("dispersion_rank", descending=True).head(self._top_n)["symbol"]]

        # Form signals
        long_positions = {s: 1.0 / len(selected_dispersed) for s in selected_dispersed}
        short_positions = {s: -1.0 / len(selected_compressed) for s in selected_compressed}

        weights = {**long_positions, **short_positions}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest