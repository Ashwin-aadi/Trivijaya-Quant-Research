from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality effects can arise due to annual cycles in economic or market behaviors. "
        "For example, certain stocks may see increased trading volumes and higher returns at "
        "specific times of the year, such as during festival seasons or end-of-year buying."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_counts: dict[str, int] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Count the number of days above a certain threshold as positive signals
            threshold = 1.02 * min(values)
            count_above_threshold = sum(1 for value in values[-30:] if value > threshold)

            symbol_counts[symbol] = count_above_threshold

        # Identify symbols with the most occurrences above the threshold
        top_symbols = sorted(symbol_counts.items(), key=lambda x: -x[1])[:5]
        weights = {symbol: 1.0 / len(top_symbols) for symbol, _ in top_symbols}

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest