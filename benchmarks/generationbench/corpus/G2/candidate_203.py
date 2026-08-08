from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "By combining two weakly related characteristics—volume and momentum—we aim to identify "
        "stocks that are both currently strong in price movement and experiencing high volume. This dual criteria can help filter out noise and isolate stocks with significant trading activity."
    )

    def __init__(self, momentum_window: int = 20, volume_threshold: float = 100_000) -> None:
        self._momentum_window = momentum_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window)
        if history.height < self._momentum_window:
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = set()
        momentum_scores = {}
        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol).to_dict(as_series=False)[0]
            latest_close = row["close"]
            volume_history = history.filter(pl.col("symbol") == symbol)["volume"].to_list()
            if len(volume_history) < self._momentum_window:
                continue
            recent_volume_mean = sum(volume_history[-self._momentum_window:]) / self._momentum_window
            if recent_volume_mean > self._volume_threshold:
                high_volume_symbols.add(symbol)
            
            # Calculate momentum score as the ratio of latest close to 20-day average
            daily_closes = [float(v) for v in row["close"].to_list()]
            mean_close = sum(daily_closes[-self._momentum_window:]) / self._momentum_window
            momentum_score = (latest_close - mean_close) / mean_close if mean_close != 0 else 0
            momentum_scores[symbol] = momentum_score

        top_momentum_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        final_picks = [s for s, _ in top_momentum_symbols if s in high_volume_symbols]

        if not final_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(final_picks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in final_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest