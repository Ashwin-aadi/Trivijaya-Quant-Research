from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "This strategy identifies strong price breakouts from support or resistance levels followed by an initial retracement. "
        "If the price subsequently moves back towards the breakout level but fails to retrace fully, it suggests momentum and can lead to continued movement in the breakout direction."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals = []
        for symbol in view.symbols:
            symbols_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )

            # Find valid breakouts
            if (symbols_history["close"].last() > symbols_history["high"][-2]
                    and symbols_history["close"][-1] < symbols_history["high"][-2]):
                breakout_signals.append({"symbol": symbol, "direction": "up"})
            elif (symbols_history["close"].last() < symbols_history["low"][-2]
                  and symbols_history["close"][-1] > symbols_history["low"][-2]):
                breakout_signals.append({"symbol": symbol, "direction": "down"})

        if not breakout_signals:
            return Signal(information_available_at=stamp, weights={})

        # Calculate retracement levels
        retracement_levels = []
        for signal in breakout_signals:
            symbol = signal["symbol"]
            direction = signal["direction"]

            if direction == "up":
                price_level = symbols_history["high"][-2]
                retracement_382 = round(price_level * 0.618, 2)
                retracement_50 = round(price_level * 0.5, 2)

                # Check for retracement
                if (symbols_history.filter(
                        pl.col("session_date") > symbols_history["session_date"].max() - pl.duration(days=1))
                    .filter(pl.col("low").lt(retracement_382)).height == 0
                    or symbols_history.filter(
                        pl.col("session_date") > symbols_history["session_date"].max() - pl.duration(days=1))
                    .filter(pl.col("low").lt(retracement_50)).height == 0):
                    retracement_levels.append({"symbol": symbol, "level": max(retracement_382, retracement_50)})

            elif direction == "down":
                price_level = symbols_history["low"][-2]
                retracement_382 = round(price_level * 1.618, 2)
                retracement_50 = round(price_level * 2, 2)

                # Check for retracement
                if (symbols_history.filter(
                        pl.col("session_date") > symbols_history["session_date"].max() - pl.duration(days=1))
                    .filter(pl.col("high").gt(retracement_382)).height == 0
                    or symbols_history.filter(
                        pl.col("session_date") > symbols_history["session_date"].max() - pl.duration(days=1))
                    .filter(pl.col("high").gt(retracement_50)).height == 0):
                    retracement_levels.append({"symbol": symbol, "level": min(retracement_382, retracement_50)})

        if not retracement_levels:
            return Signal(information_available_at=stamp, weights={})

        # Rank by volume and select top N
        ranked_signals = sorted(
            [(signal["symbol"], signal["level"]) for signal in retracement_levels],
            key=lambda x: float(view.history().filter(pl.col("symbol") == x[0])[-1]["volume"]),
            reverse=True,
        )[: self._top_n]

        weights = {symbol: 1.0 / len(ranked_signals) for symbol, _ in ranked_signals}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest