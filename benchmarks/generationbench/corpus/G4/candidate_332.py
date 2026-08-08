from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying stocks with "
        "significant increases in trading volume and confirming price movements. High volume "
        "indicates increased investor interest and potential significant price changes."
    )

    def __init__(self, window: int = 1) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate volume change
        latest_close = view.latest_close()
        prev_close = {symbol: pl.DataFrame(history[symbol]).select("adj_close").shift_and_fill(1).to_dict(True)[0][0] for symbol in view.symbols}
        volume_change = {
            symbol: (latest_close[symbol] / prev_close[symbol] - 1.0) * 100
            for symbol in latest_close if symbol in prev_close and prev_close[symbol] > 0
        }
        
        # Filter symbols with significant volume increase (>50%)
        high_volume_symbols = {symbol: vol_change for symbol, vol_change in volume_change.items() if vol_change >= 50}

        # Confirm price direction based on closing prices
        price_direction_confirmation = {
            symbol: (latest_close[symbol] > max(history[symbol]["high"])) if vol_change >= 50 else False
            for symbol, vol_change in high_volume_symbols.items()
        }

        filtered_high_volume_symbols = {symbol for symbol, confirmed in price_direction_confirmation.items() if confirmed}

        if not filtered_high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_high_volume_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in filtered_high_volume_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest