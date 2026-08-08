from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: short-term momentum and "
        "volume anomaly. Short-term momentum suggests that assets with high returns over the "
        "last few days may continue to outperform, while volume anomalies indicate that "
        "abnormal trading volumes could signal potential reversals or continuation of trends."
    )

    def __init__(self, window: int = 5, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            # Calculate short-term momentum score
            last_close = adj_closes[-1]
            prev_close = adj_closes[-2]
            momentum_score = (last_close - prev_close) / prev_close
            momentum_scores[symbol] = momentum_score

        volume_anomalies: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in view.history().columns:
                continue
            hist = view.history(symbol=symbol)
            latest_volume = float(hist.select("volume").max().item())
            prev_volume = float(hist.select("volume").shift(1).max().item())

            # Calculate volume anomaly score
            volume_anomaly_score = latest_volume / prev_volume if prev_volume > 0 else 0.0
            volume_anomalies[symbol] = volume_anomaly_score

        combined_scores: dict[str, float] = {
            symbol: (momentum_scores.get(symbol) or 0.0) + (volume_anomalies.get(symbol) or 0.0)
            for symbol in view.symbols
        }

        if not combined_scores:
            return Signal(information_available_at=stamp, weights={})

        # Sort by combined score and pick top N symbols
        sorted_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest