from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrendFollow(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies aim to capture trends while "
        "adjusting the position size based on recent volatility. High volatility can "
        "indicate that a new trend is forming or that price action is becoming more "
        "jittery, leading to larger positions in trending markets."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_positions: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate daily returns
            daily_returns = (pl.Series(values).shift_and_fill(1, 0) / pl.Series(values) - 1.0).to_list()[1:]

            # Calculate rolling volatility
            rolling_volatility = (pl.Series(daily_returns).rolling_std(window_length=self._window)).to_list()

            # Trend direction: positive or negative
            trend_direction = "bullish" if daily_returns[-1] > 0 else "bearish"

            # Position size based on recent volatility
            position_size = self._threshold * rolling_volatility[-1]

            volatility_scaled_positions[symbol] = {
                "trend": trend_direction,
                "volatility": rolling_volatility[-1],
                "position_size": position_size
            }

        if not volatility_scaled_positions:
            return Signal(information_available_at=stamp, weights={})

        # Determine the net position based on trends and volatility
        bullish_symbols = {k: v["position_size"] for k, v in volatility_scaled_positions.items() if v["trend"] == "bullish"}
        bearish_symbols = {k: -v["position_size"] for k, v in volatility_scaled_positions.items() if v["trend"] == "bearish"}

        net_position = sum(bullish_symbols.values()) + sum(bearish_symbols.values())
        weight_per_symbol = 1.0 / len(volatility_scaled_positions) if net_position != 0 else 0

        return Signal(
            information_available_at=stamp,
            weights={
                s: (volatility_scaled_positions[s]["position_size"] if volatility_scaled_positions[s]["trend"] == "bullish" else -volatility_scaled_positions[s]["position_size"]) * weight_per_symbol
                for s in view.symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest