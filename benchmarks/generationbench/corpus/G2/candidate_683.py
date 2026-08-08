from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "By combining volume-based momentum with earnings surprise, we aim to identify "
        "stocks that not only have strong recent trading volumes but also positive earnings "
        "surprises. This dual characteristic often indicates a stock is undervalued and due for "
        "a price increase."
    )

    def __init__(self, volume_window: int = 10, eps_window: int = 5) -> None:
        self._volume_window = volume_window
        self._eps_window = eps_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._volume_window, self._eps_window))
        if history.height < max(self._volume_window, self._eps_window):
            return Signal(information_available_at=stamp, weights={})

        volume_signals: dict[str, float] = {}
        eps_signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            # Volume-based signal
            volumes = history.filter(pl.col("symbol") == symbol)[
                "volume"
            ].drop_nulls().to_list()
            volume_signal = (
                pl.DataFrame([{"symbol": symbol, "volume": v} for v in volumes])
                .group_by("symbol")
                .agg((pl.col("volume").mean() / pl.col("volume").shift(1) - 1.0).alias("v_mean_ratio"))
                .sort("v_mean_ratio", descending=True)
                .select("symbol")
            )
            volume_signals[symbol] = float(volume_signal[0]["symbol"]) if not volume_signal.is_empty() else 0.0

            # EPS-based signal
            eps_history = history.filter(pl.col("symbol") == symbol).sort("session_date").tail(self._eps_window)
            eps_signal = (
                eps_history.with_columns(
                    (pl.col("adj_close") - pl.col("adj_close").shift(1)).alias("eps_change")
                ).group_by("symbol")
                .agg((pl.col("eps_change").mean() / pl.col("eps_change").shift(1) - 1.0).alias("eps_mean_ratio"))
                .sort("eps_mean_ratio", descending=True)
                .select("symbol")
            )
            eps_signals[symbol] = float(eps_signal[0]["symbol"]) if not eps_signal.is_empty() else 0.0

        combined_scores = {s: (volume_signals[s] + eps_signals[s]) for s in view.symbols}
        sorted_symbols = sorted(combined_scores, key=combined_scores.get, reverse=True)
        top_symbols = sorted_symbols[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest