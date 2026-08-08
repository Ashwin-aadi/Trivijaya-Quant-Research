from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityReversion(Strategy):
    rationale = (
        "This strategy exploits the tendency of markets to revert after periods of high price "
        "volatility. It identifies stocks with recent high volatility and waits for signs of range "
        "compression before entering long positions on the lower end of the compressed range, "
        "anticipating a potential upward move back towards mean levels."
    )

    def __init__(self, atr_window: int = 20, compression_window: int = 5) -> None:
        self._atr_window = atr_window
        self._compression_window = compression_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._atr_window + self._compression_window)
        if history.height < self._atr_window + self._compression_window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Step 1: Compute ATR
        def atr(df: pl.DataFrame) -> float:
            hlc = df.select(
                pl.col("high").max().alias("h"),
                pl.col("low").min().alias("l"),
                pl.col("close").shift(-1).alias("c")
            ).select((pl.col("h") - pl.col("l")).mean() + (pl.col("h") - pl.col("c")).abs().mean())
            return float(atr.to_list()[0])

        atr_df = history.select(
            pl.all().exclude("session_date").apply(atr, name="atr")
        )
        atr_series = atr_df["atr"].to_list()

        # Step 2: Identify high dispersion stocks
        top_20_percent = int(len(symbols) * 0.2)
        sorted_symbols = [s for s in symbols if f"{s}_atr" in atr_series.columns]
        top_dispersion = [
            symbol for _, symbol in sorted(zip(atr_series, sorted_symbols), reverse=True)[:top_20_percent]
        ]

        # Step 3: Monitor volatility compression
        adrs = view.history(lookback=self._compression_window).select(
            pl.all().exclude("session_date").apply(lambda x: (x.max() - x.min()).mean(), name="adr")
        )
        adrs_series = [float(v) for v in adrs["adr"].to_list()]
        avg_adr = sum(adrs_series[-self._compression_window:]) / self._compression_window
        if all(a <= 0.8 * avg_adr for a in adrs_series):
            # Step 4: Entry signal on range compression
            compressed_symbols = [
                symbol for _, symbol in sorted(zip(atr_series, top_dispersion), reverse=False)
            ][:5]
            weight = 1.0 / len(compressed_symbols)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in compressed_symbols}
            )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest