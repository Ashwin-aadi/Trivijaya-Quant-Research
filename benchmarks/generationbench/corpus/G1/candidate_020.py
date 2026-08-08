from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong momentum "
        "and can signal potential profit opportunities. This strategy identifies stocks "
        "that have experienced a significant price increase or decrease on high volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {symbol: [] for symbol in view.symbols}
        for _, row in history.filter(pl.col("session_date") < stamp).rows():
            for symbol in view.symbols:
                price = row[symbol]
                if not pl.isnan(price):
                    symbol_prices[symbol].append(float(price))

        signals: list[str] = []
        for symbol, prices in symbol_prices.items():
            if len(prices) < self._window - 1:
                continue

            last_price = prices[-1]
            prev_close = prices[-2]

            volume_change = (last_price - prev_close) / abs(prev_close)
            high_volume = max(view.closes().get_column(symbol).to_list()) > view.latest_close()[symbol] * 1.5

            if volume_change > 0 and high_volume:
                signals.append(symbol)
            elif volume_change < 0 and high_volume:
                signals.append(f"-{symbol}")

        weight = 1.0 / len(signals) if signals else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest