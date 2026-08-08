from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large trading volumes often precede significant price movements. This strategy "
        "identifies stocks with sudden volume surges and enters positions based on the "
        "direction of subsequent price changes."
    )

    def __init__(self, volume_window: int = 30 * 6, price_window: int = 10) -> None:
        self._volume_window = volume_window
        self._price_window = price_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._volume_window + self._price_window - 1)

        if history.is_empty() or history.height < self._volume_window + self._price_window - 1:
            return Signal(information_available_at=stamp, weights={})

        # Compute volume and price metrics
        volume_metrics = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
                (pl.col("volume") / pl.col("adj_close").shift(1) - 1).alias("vol_change_rate"),
                pl.col("close").shift(-self._price_window + 1).first().alias("latest_close"),
            )
        )

        # Filter symbols with sufficient history
        symbols = [s for s in view.symbols if s in volume_metrics.columns]
        filtered_history = history.select(["symbol", "session_date"] + symbols)

        # Calculate volume anomaly score and price change percentage
        volume_anomaly_score = (
            pl.concat_frames(
                [
                    volume_metrics.filter(pl.col("symbol") == s).with_columns(
                        (pl.col("volume").mean() / pl.col("avg_volume")) * 1.5 - 1.0
                    ).select([pl.col("symbol"), "vol_change_rate"])
                    for s in symbols
                ]
            )
            .filter(pl.col("symbol").is_in(symbols))
            .sort("symbol")
        )

        price_change = (
            filtered_history.filter(
                pl.col("session_date") >= stamp - date(0, 0, self._price_window)
            )
            .group_by("symbol")
            .agg(
                (pl.col("close").last() / pl.col("latest_close") - 1).alias("price_change")
            )
        )

        # Combine metrics and rank
        combined = (
            volume_anomaly_score.join(price_change, on="symbol", how="inner")
            .with_columns((pl.col("vol_change_rate") + pl.col("price_change")).alias("score"))
            .sort(by="score", descending=True)
            .select(["symbol"])
        )

        top_symbols = combined.head(20)["symbol"].to_list()
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.05 / len(top_symbols)
        weights = {s: weight for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest