from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy exploits significant price movements that are accompanied by increased "
        "trading volume. It identifies stocks showing both substantial price changes and higher "
        "than normal trading activity, indicating potential directional moves driven by strong news or events."
    )

    def __init__(self, lookback_period: int = 30, z_score_threshold: float = 1.5) -> None:
        self._lookback_period = lookback_period
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        volume_z_scores: dict[str, float] = {}
        price_changes: dict[str, float] = {}

        for symbol in view.symbols:
            closes = [float(v) for v in view.closes(lookback=self._lookback_period)[symbol].to_list()]
            if len(closes) < self._lookback_period - 1:
                continue
            open_price = history.select(pl.col("symbol") == symbol).select("open").item()
            close_price = closes[-1]
            price_change = (close_price - open_price) / open_price

            if abs(price_change) >= 0.02:  # 2% threshold for significant move
                volume_series = [float(v) for v in history.select(pl.col("symbol") == symbol).select("volume").to_list()[0]]
                mean_volume = pl.Series(volume_series).mean().item()
                z_score = (max(volume_series) - mean_volume) / mean_volume

                if z_score > self._z_score_threshold:
                    volume_z_scores[symbol] = z_score
                    price_changes[symbol] = price_change

        ranked_signals = sorted(
            [(symbol, price_changes[symbol], volume_z_scores[symbol]) for symbol in price_changes],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        if not ranked_signals:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [signal[0] for signal in ranked_signals[:5]]  # Top 5 signals
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest