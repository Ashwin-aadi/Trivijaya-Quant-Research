from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion30d(Strategy):
    rationale = (
        "The strategy identifies stocks that have deviated significantly from their 30-day "
        "moving average, aiming to capitalize on short-term mean reversion. It uses z-scores to "
        "select top and bottom performers for long and short positions respectively."
    )

    def __init__(self, window: int = 30, z_score_threshold_long: float = 1.5,
                 z_score_threshold_short: float = -1.5) -> None:
        self._window = window
        self._z_score_threshold_long = z_score_threshold_long
        self._z_score_threshold_short = z_score_threshold_short

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [sym for sym in view.symbols if sym in history.symbol.unique().to_list()]
        z_scores_long: dict[str, float] = {}
        z_scores_short: dict[str, float] = {}

        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            close_prices = [float(v) for v in df["adj_close"].to_list()]
            ma_30 = sum(close_prices[-self._window:]) / self._window
            std_30 = (sum((p - ma_30) ** 2 for p in close_prices[-self._window:]) / self._window) ** 0.5

            z_score_long = (close_prices[-1] - ma_30) / std_30 if std_30 != 0 else 0
            z_scores_long[symbol] = z_score_long

            z_score_short = -(close_prices[-1] - ma_30) / std_30 if std_30 != 0 else 0
            z_scores_short[symbol] = z_score_short

        long_signals: list[str] = [s for s, z in z_scores_long.items() if z > self._z_score_threshold_long]
        short_signals: list[str] = [s for s, z in z_scores_short.items() if z < self._z_score_threshold_short]

        long_signals_top_20_percent = sorted(long_signals, key=lambda x: z_scores_long[x], reverse=True)[:int(0.2 * len(long_signals))]
        short_signals_top_20_percent = sorted(short_signals, key=lambda x: z_scores_short[x], reverse=False)[:int(0.2 * len(short_signals))]

        long_positions = {s: 0.05 for s in long_signals_top_20_percent}
        short_positions = {s: -0.05 for s in short_signals_top_20_percent}

        if not (long_positions or short_positions):
            return Signal(information_available_at=stamp, weights={})

        combined_weights = {**long_positions, **short_positions}
        return Signal(
            information_available_at=stamp,
            weights=combined_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest