from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to have accurate and stable price discovery. "
        "By equally weighting these highly liquid stocks, we can aim for a diversified portfolio "
        "that benefits from the reduced idiosyncratic risk associated with less-liquid securities."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter symbols based on liquidity criteria
        filtered_symbols = _filter_by_liquidity(history).to_dict(True)
        
        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting the selected symbols
        weight = 1.0 / len(filtered_symbols)
        weights = {symbol: weight for symbol in filtered_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _filter_by_liquidity(history: pl.DataFrame) -> pl.Series:
    symbols = history["symbol"].to_list()
    filtered_symbols = []
    
    for symbol in symbols:
        latest_volume = float(history.filter(pl.col("symbol") == symbol)["volume"].max().item())
        if latest_volume > 1000000:  # Arbitrary threshold for high liquidity
            filtered_symbols.append(symbol)
    
    return pl.Series(symbols=filtered_symbols, dtype=pl.Utf8)