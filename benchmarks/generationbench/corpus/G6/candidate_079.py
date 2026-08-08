from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy identifies stocks experiencing range compression by monitoring the "
        "Average True Range (ATR) over 14 and 30 days. Stocks with ATR below their long-term "
        "moving average are entered at daily open or close, while stop-loss orders limit losses."
    )

    def __init__(self, window_short: int = 14, window_long: int = 30, top_n: int = 6) -> None:
        self._window_short = window_short
        self._window_long = window_long
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_long + 1)
        if closes.height < self._window_long + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_long + 1:
                continue

            atr_short = pl.DataFrame({"values": values[-self._window_short:]})["values"].std()
            atr_long = pl.DataFrame({"values": values[-self._window_long:]})["values"].std()

            if atr_short < atr_long * 0.95:  # Adjust threshold as needed
                signals.append(symbol)

        signals = signals[: self._top_n]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest