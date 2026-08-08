from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken through key resistance levels "
        "with increased trading volumes, indicating strong market sentiment. It uses Bollinger Bands "
        "confirmation and equal dollar-weighted positions to balance risk management and diversification."
    )

    def __init__(self, window: int = 50, volume_increase_threshold: float = 1.5) -> None:
        self._window = window
        self._volume_increase_threshold = volume_increase_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            open_prices = [float(v) for v in history[history["symbol"] == symbol]["open"].to_list()]
            close_prices = [float(v) for v in history[history["symbol"] == symbol]["close"].to_list()]
            volume = [float(v) for v in history[history["symbol"] == symbol]["volume"].to_list()]

            if len(open_prices) < self._window:
                continue

            resistance_level = max(close_prices[-20:])
            last_close = close_prices[-1]
            last_volume = volume[-1]

            if last_close > resistance_level and last_volume >= last_volume * self._volume_increase_threshold:
                picks.append(symbol)

        picks = picks[:10]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.3 / len(picks)
        signal = Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )

        # Bollinger Bands Confirmation
        history_bbands = view.history(lookback=self._window)
        bbands_confirmed = False
        for symbol in picks:
            close_prices_bb = [float(v) for v in history_bbands[history_bbands["symbol"] == symbol]["close"].to_list()]
            if len(close_prices_bb) < self._window:
                continue

            mid_price = pl.DataFrame({"close": close_prices_bb}).with_columns((pl.col("close").rolling_mean(window_size=self._window, center=True)).alias("mid"))
            std_dev = (pl.DataFrame({"close": close_prices_bb}).with_columns((pl.col("close") - pl.col("mid")).abs().rolling_mean(window_size=self._window, center=True)).with_columns((pl.col("close") - pl.col("mid")) / 2.0).with_columns(pl.col("close").std()).alias("std"))
            upper_band = (mid_price["mid"] + 2 * std_dev["std"]).to_list()
            lower_band = (mid_price["mid"] - 2 * std_dev["std"]).to_list()

            if last_close > max(upper_band) or last_close < min(lower_band):
                bbands_confirmed = True
                break

        if not bbands_confirmed:
            return Signal(information_available_at=stamp, weights={})

        return signal


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest