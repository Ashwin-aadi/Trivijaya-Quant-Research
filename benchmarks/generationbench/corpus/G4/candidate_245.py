from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategy exploits the tendency for prices that have broken "
        "through a significant resistance or support level to continue moving in the direction "
        "of the breakout. This strategy aims to capture momentum after a breakout while managing risk."
    )

    def __init__(self, window: int = 30, rank_threshold: float = 1.2, stop_loss_pct: float = 0.1) -> None:
        self._window = window
        self._rank_threshold = rank_threshold
        self._stop_loss_pct = stop_loss_pct

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            closes = df.select(pl.col("adj_close")).to_numpy().flatten()
            open_price = float(df.select(pl.col("open")).to_numpy()[0][0])
            high_price = float(df.select(pl.col("high")).to_numpy()[0][0])
            low_price = float(df.select(pl.col("low")).to_numpy()[0][0])
            volume = int(df.select(pl.col("volume")).to_numpy()[0][0])

            if df.shape[0] < self._window:
                continue

            last_close = closes[-1]
            previous_high = max(closes[:-1])
            previous_low = min(closes[:-1])

            # Calculate breakout percentage
            if open_price > high_price:
                breakout_direction = "up"
                resistance_level = previous_high
            else:
                breakout_direction = "down"
                support_level = previous_low

            if (breakout_direction == "up" and last_close > resistance_level) or \
               (breakout_direction == "down" and last_close < support_level):
                # Check for volume confirmation
                avg_volume = df.select(pl.col("volume").mean()).to_numpy()[0][0]
                breakout_volume = volume / avg_volume

                if breakout_volume >= 1.5:  # Adjust this threshold as needed
                    rank_score = (last_close - previous_high) / (previous_high - previous_low) \
                                 if breakout_direction == "up" else \
                                 (previous_low - last_close) / (previous_high - previous_low)

                    if rank_score >= self._rank_threshold:
                        breakout_signals[symbol] = rank_score * breakout_volume

        if not breakout_signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(breakout_signals.values())
        weights = {symbol: weight / total_weight for symbol, weight in breakout_signals.items()}
        top_n_symbols = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
        return Signal(information_available_at=stamp, weights=dict(top_n_symbols))


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest