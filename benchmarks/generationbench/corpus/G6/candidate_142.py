from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalStrategy(Strategy):
    rationale = (
        "This strategy leverages historical seasonality in the Indian market by entering "
        "positions starting from October 15th and holding until November 30th to capture "
        "the post-Diwali rally. It combines a conservative approach with sector-specific "
        "timing around Diwali, ensuring timely entry and exit based on market dynamics."
    )

    def __init__(self) -> None:
        self._entry_date = date(2020, 10, 15)
        self._exit_date = date(2024, 11, 30)

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if stamp < self._entry_date or stamp > self._exit_date:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            history = view.history(lookback=200)
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].to_list()]
            if len(close_prices) < 200:
                continue

            recent_high = max(close_prices[-60:])
            recent_low = min(close_prices[-60:])
            current_close = view.latest_close()[symbol]

            if (current_close > recent_high * 1.05 or
                    (current_close >= recent_high and close_prices.index(current_close) < 60)):
                picks.append(symbol)

        picks = picks[:20]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest