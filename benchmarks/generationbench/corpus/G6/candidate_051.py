from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualStrategy(Strategy):
    rationale = (
        "This strategy aims to balance simplicity with robust risk management by selecting and "
        "equally weighting stocks based on their liquidity. It ensures sufficient trading volume "
        "and low bid-ask spreads, reducing turnover while capturing market returns."
    )

    def __init__(self, min_volume: float = 0.5e6, max_names: int = 100) -> None:
        self._min_volume = min_volume
        self._max_names = max_names

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365 * 2)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = history.select(
            pl.col("symbol"), pl.col("volume").sum().alias("total_volume")
        )
        filtered_symbols = [
            symbol
            for (symbol, total_volume) in volume_df.to_dicts()
            if total_volume > self._min_volume
        ]

        if len(filtered_symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=365)
        symbols_with_closes = [symbol for symbol in filtered_symbols if symbol in closes.columns]

        if len(symbols_with_closes) <= self._max_names:
            names = symbols_with_closes
        else:
            names = sorted(
                symbols_with_closes, key=lambda x: -float(closes[x].tail(1).item())
            )[: self._max_names]

        weight = 1.0 / len(names)
        return Signal(
            information_available_at=stamp,
            weights={name: weight for name in names},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest