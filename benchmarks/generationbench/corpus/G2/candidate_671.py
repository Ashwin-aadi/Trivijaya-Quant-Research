from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeRSIAndVolume(Strategy):
    rationale = (
        "The combination of Relative Strength Index (RSI) and volume can provide insights into "
        "the strength and sustainability of a trend. High RSI levels in conjunction with high "
        "volume suggest strong buying pressure, while low RSI levels with increasing volume "
        "indicate potential for a turnaround."
    )

    def __init__(self, rsi_window: int = 14, volume_threshold: float = 100_000) -> None:
        self._rsi_window = rsi_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._rsi_window + 1).sort("session_date")
        if history.height < self._rsi_window + 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        volumes = view.history()[["symbol", "volume"]].filter(
            pl.col("session_date").is_in(history.select("session_date").to_list())
        ).sort("session_date")

        rsi_values = _calculate_rsi(closes, self._rsi_window)
        high_volume_symbols = set(volumes.filter(pl.col("volume") > self._volume_threshold)["symbol"].to_list())

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in rsi_values.columns or symbol not in volumes["symbol"].to_list():
                continue
            if rsi_values[symbol][-1] >= 70 and symbol in high_volume_symbols:
                picks.append(symbol)
            elif rsi_values[symbol][-1] <= 30 and symbol in high_volume_symbols:
                picks.append(symbol)

        picks = list(set(picks))[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_rsi(frame: pl.DataFrame, window: int) -> pl.DataFrame:
    delta = frame.with_columns((pl.col("close") - pl.col("close").shift(1)).alias("delta"))
    gain = (delta.filter(pl.col("delta") > 0).with_columns((pl.col("delta") / 2.0).alias("gain"))).group_by("symbol").agg(
        pl.col("gain").mean().alias("avg_gain")
    )
    loss = (delta.filter(pl.col("delta") < 0).with_columns((-1 * pl.col("delta") / 2.0).alias("loss"))).group_by("symbol").agg(
        pl.col("loss").mean().alias("avg_loss")
    )

    rsi = gain.join(loss, on="symbol", how="inner").with_columns(
        (100 - (100 / ((pl.col("avg_gain") + 1) / (pl.col("avg_loss") + 1)))).alias("rsi")
    )
    return rsi