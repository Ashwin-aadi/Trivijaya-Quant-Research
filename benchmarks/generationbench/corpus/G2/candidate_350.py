from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility can indicate a market is in a trend. By focusing on symbols with high "
        "recent volatility, we can identify potential strong trends and capitalize on them. "
        "This strategy aims to outperform during periods of trending markets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_ratios: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate daily returns and their standard deviation (volatility)
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            volatility = pl.DataFrame({"returns": returns}).select(pl.col("returns").std().alias("volatility"))[0, 0]
            volatility_ratio = volatility / (max(values) - min(values))

            if not pl.is_nan(volatility_ratio):
                volatility_ratios.append((symbol, volatility_ratio))

        # Filter out the symbols with the highest volatility ratios
        top_symbols = [s for s, _ in sorted(volatility_ratios, key=lambda x: x[1], reverse=True)[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest