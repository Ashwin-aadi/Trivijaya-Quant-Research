from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedAmbitiousBreakoutContinuation(Strategy):
    rationale = (
        "Identify stocks breaking through support or resistance levels using daily OHLC data and volume to confirm breakouts. "
        "Enter positions when a stock closes above its 20-day moving average after sustained price movement beyond this level for at least one day, and confirm on the third consecutive trading session where the stock remains above the breakout level."
        "Exit if the stock’s price falls below its 10-day moving average or rises above its 30-day moving average, or after up to 7 days from entry. Maintain a balanced portfolio of 15-20 stocks with equal initial weights and implement stop-loss orders to protect against significant drawdowns."
    )

    def __init__(self, window_breakout: int = 20, window_ema_10: int = 10, window_ema_30: int = 30, days_for_entry: int = 3, max_days_in_portfolio: int = 7) -> None:
        self._window_breakout = window_breakout
        self._window_ema_10 = window_ema_10
        self._window_ema_30 = window_ema_30
        self._days_for_entry = days_for_entry
        self._max_days_in_portfolio = max_days_in_portfolio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_breakout, self._window_ema_10, self._window_ema_30))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            close_hist = [float(v) for v in history.select("adj_close").filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            ema_20 = _rolling_mean(close_hist[-self._window_breakout:], self._window_breakout)
            ema_10 = _rolling_mean(close_hist[-self._window_ema_10:], self._window_ema_10)
            ema_30 = _rolling_mean(close_hist[-self._window_ema_30:], self._window_ema_30)

            if len(ema_20) < self._days_for_entry:
                continue

            entry_conditions_met = False
            for i in range(self._days_for_entry - 1, len(close_hist)):
                if close_hist[i] > ema_20[-1]:
                    if all(close_hist[j] > ema_20[-1] for j in range(i, min(i + self._days_for_entry, len(close_hist)))):
                        entry_conditions_met = True
                        break

            if not entry_conditions_met:
                continue

            exit_conditions_met = False
            for i in range(len(close_hist) - 1, max(0, len(close_hist) - self._max_days_in_portfolio - 1), -1):
                if (close_hist[i] < ema_10[-1]) or (close_hist[i] > ema_30[-1]):
                    exit_conditions_met = True
                    break

            if exit_conditions_met:
                continue

            picks.append(symbol)

        picks = picks[:20]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.05 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest


def _rolling_mean(values: list[float], window: int) -> list[float]:
    mean_values = [sum(values[:i+1]) / (i + 1) for i in range(window)]
    rolling_means = []
    for i in range(len(values) - window + 1):
        rolling_means.append(sum(values[i:i+window]) / window)
    return rolling_means