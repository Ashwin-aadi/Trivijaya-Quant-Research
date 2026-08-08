from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "This strategy aims to capitalize on periods of market dispersion or range compression "
        "in Indian equities. It enters long positions during range compression and short positions "
        "during high dispersion, leveraging the tendency for markets to reverse after extreme "
        "conditions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate Bollinger Bands
        bb_bands = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") + pl.col("low") + 2 * pl.col("close")) / 4,
                (pl.col("high") - pl.col("low")) / 2
            )
            .with_columns(
                [
                    (pl.col("close").rolling_mean(self._window)).alias(f"bb_middle"),
                    ((2 * pl.col("close")).rolling_std(self._window)).alias(f"bb_deviation")
                ]
            )
            .with_column(
                (pl.col(f"bb_middle") + 2 * pl.col(f"bb_deviation")).alias(f"bb_upper")
            )
            .with_column((pl.col(f"bb_middle") - 2 * pl.col(f"bb_deviation")).alias(f"bb_lower"))
        )

        # Calculate ATR
        atr = (
            history.select(
                pl.col("symbol"),
                (pl.col("high").shift(1) - pl.col("low").shift(1)).abs(),
                ((pl.col("high").shift(1) - pl.col("close").shift(1)).abs()),
                ((pl.col("low").shift(1) - pl.col("close").shift(1)).abs())
            )
            .with_column(
                (pl.max(pl.all(), skip_nulls=True)).alias(f"tr")
            )
            .with_column((pl.col(f"tr").rolling_mean(self._window)).alias(f"atr"))
        )

        # Combine Bollinger Bands and ATR
        combined = bb_bands.join(atr, on="symbol", how="inner")

        # Rank stocks based on dispersion or compression
        dispersion_rank = (
            combined.select(
                pl.col("symbol"),
                (pl.col(f"bb_upper") - pl.col(f"bb_lower")).alias(f"band_width")
            )
            .with_column((pl.col(f"atr").rank(method="ordinal", descending=True)).alias(f"atr_rank"))
            .sort("atr_rank", descending=True)
            .head(self._window)
        )

        compression_symbols = [symbol for symbol, _ in dispersion_rank.rows()]
        dispersion_symbols = [
            symbol for symbol, width in dispersion_rank.sort(f"band_width").rows()
            if symbol not in compression_symbols
        ]

        # Determine signals
        long_weights = {s: 1.0 / len(compression_symbols) for s in compression_symbols}
        short_weights = {s: -1.0 / len(dispersion_symbols) for s in dispersion_symbols}

        weights = {**long_weights, **short_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest