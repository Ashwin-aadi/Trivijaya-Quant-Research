from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 50-day moving average "
        "and the relative strength index (RSI) to identify potential entry points. The moving "
        "average helps filter out noise and smooth price action, while RSI indicates overbought "
        "or oversold conditions."
    )

    def __init__(self, ma_window: int = 50, rsi_lookback: int = 14) -> None:
        self._ma_window = ma_window
        self._rsi_lookback = rsi_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._rsi_lookback + max(self._ma_window, 1))
        if closes.height < self._rsi_lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        ma_signal: list[str] = []
        rsi_signal: list[str] = []

        for symbol in view.symbols:
            history = view.history(lookback=self._ma_window + max(self._rsi_lookback, 1))
            if symbol not in history.columns:
                continue

            # Calculate moving average
            ma_close = float(
                history[symbol].mean().item()
            )
            current_close = float(view.latest_close()[symbol])
            if current_close > ma_close:
                ma_signal.append(symbol)

            # Calculate RSI
            closes_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            price_changes = [(closes_series[i] - closes_series[i-1]) / abs(closes_series[i-1]) if closes_series[i-1] != 0 else 0 for i in range(1, len(closes_series))]
            gains = [change for change in price_changes if change > 0]
            losses = [-change for change in price_changes if change < 0]

            avg_gain = sum(gains) / self._rsi_lookback
            avg_loss = sum(losses) / self._rsi_lookback

            rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))
            if rsi < 30:
                rsi_signal.append(symbol)

        # Intersection of both signals
        combined_signals = list(set(ma_signal) & set(rsi_signal))

        if not combined_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in combined_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest