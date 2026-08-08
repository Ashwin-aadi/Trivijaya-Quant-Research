from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "This strategy exploits dispersion or range compression in the Indian equity markets by "
        "identifying stocks showing signs of increased volatility (dispersion) or reduced volatility "
        "(range compression). Mean-reversion trades are then executed based on these identified opportunities."
    )

    def __init__(self, dispersion_window: int = 40, atr_window: int = 14, top_n: int = 20) -> None:
        self._dispersion_window = dispersion_window
        self._atr_window = atr_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._dispersion_window + self._atr_window)

        if history.height < self._dispersion_window + self._atr_window:
            return Signal(information_available_at=stamp, weights={})

        bollinger_bands = self._calculate_bollinger_bands(history)
        atr_values = self._calculate_atr(history)

        picks: list[str] = []
        for symbol in view.symbols:
            if (symbol not in bollinger_bands.columns or
                    symbol not in atr_values.columns):
                continue

            bandwidth = float(bollinger_bands[symbol].mean().alias("bandwidth").to_list()[0])
            atr = float(atr_values[symbol].mean().alias("atr").to_list()[0])

            composite_score = 1.0 / (bandwidth + atr)
            picks.append((symbol, composite_score))

        picks.sort(key=lambda x: x[1], reverse=True)
        top_n_picks = [p[0] for p in picks[:self._top_n]]
        if not top_n_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_picks}
        )

    def _calculate_bollinger_bands(self, history: pl.DataFrame) -> pl.DataFrame:
        df = history.select(
            pl.col("symbol"),
            (pl.col("adj_close") - pl.col("adj_close").rolling_mean(window_size=self._dispersion_window)).alias("mean_deviation")
        ).with_columns(
            (pl.col("mean_deviation") / (2 * pl.col("adj_close").rolling_std(window_size=self._dispersion_window))).alias("bb")
        )
        return df.group_by("symbol").agg([
            pl.col("bb").max().alias("upper_band"),
            pl.col("bb").min().alias("lower_band"),
            ((pl.col("bb").max() - pl.col("bb").min()).abs() / 2).alias("bandwidth")
        ])

    def _calculate_atr(self, history: pl.DataFrame) -> pl.DataFrame:
        df = history.select(
            pl.col("symbol"),
            (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().alias("tr")
        ).with_columns(
            ((pl.col("tr") + pl.col("high") - pl.col("low")).max()).alias("atr")
        )
        return df.group_by("symbol").agg([
            (pl.col("atr").rolling_mean(window_size=self._atr_window)).alias("atr")
        ])


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest