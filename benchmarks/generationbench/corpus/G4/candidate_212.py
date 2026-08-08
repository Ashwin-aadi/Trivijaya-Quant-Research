from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy captures volume-confirmed directional moves in the Indian equity market "
        "by leveraging the relationship between trading volume and price trends. High volumes often"
        "precede significant price movements due to increased investor participation and sentiment."
    )

    def __init__(self, volatility_window: int = 20, volume_change_window: int = 5, avg_volume_window: int = 30) -> None:
        self._volatility_window = volatility_window
        self._volume_change_window = volume_change_window
        self._avg_volume_window = avg_volume_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._volatility_window + self._volume_change_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._volatility_window)
        volumes = view.history().select("symbol", "session_date", "volume")

        # Calculate log returns and volatility
        price_returns = (history["close"] / history["adj_close"].shift(1)) - 1.0
        volatilities = (
            price_returns.to_frame(name="return")
            .group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .select(["symbol", "volatility"])
        )

        # Calculate volume change percentage
        avg_volume = volumes.groupby("symbol").agg(
            (pl.col("volume").mean().alias(f"avg_volume"))
        )
        recent_volumes = view.history(lookback=self._avg_volume_window).group_by("symbol").agg(
            pl.col("volume").max().alias("recent_volume")
        )
        volume_changes = (
            avg_volume.join(recent_volumes, on="symbol", how="inner")
            .with_columns(
                (pl.col(f"recent_volume") / pl.col(f"avg_volume") - 1.0).alias("volume_change_percent")
            )
            .select(["symbol", "volume_change_percent"])
        )

        # Identify price direction
        uptrend_symbols = []
        for symbol in symbols:
            if symbol not in closes.columns or symbol not in volume_changes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._volatility_window - 1:
                continue
            price_trend = (values[-1] > max(values[:-self._volume_change_window])) and (
                any(
                    (values[i + 1] >= values[i]) for i in range(len(values) - self._volume_change_window)
                )
            )
            if price_trend:
                uptrend_symbols.append(symbol)

        # Combine all criteria
        filtered_symbols = []
        for symbol in symbols:
            if (
                symbol not in volatilities.columns
                or symbol not in volume_changes.columns
                or symbol not in uptrend_symbols
            ):
                continue

            volatility = float(volatilities[volatilities["symbol"] == symbol]["volatility"])
            volume_change_percent = float(volume_changes[volume_changes["symbol"] == symbol]["volume_change_percent"])

            if (
                volatility < volatilities.select("volatility").quantile(0.25).item()
                and volume_change_percent > 0.5
            ):
                filtered_symbols.append(symbol)

        top_n_symbols = sorted(filtered_symbols, key=lambda s: float(closes[symbol][-1]), reverse=True)[:3]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest