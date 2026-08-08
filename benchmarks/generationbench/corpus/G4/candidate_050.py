from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the relationship between market trends and volatility in Indian equity markets. "
        "High volatility often precedes trend reversals or consolidations, while low volatility indicates sustained trending conditions. "
        "By using a volatility indicator (ATR) to scale trend-following signals, we can increase position size during low volatility periods for strong trends and reduce it during high volatility periods to mitigate risk."
    )

    def __init__(self, sma_window: int = 20, atr_window: int = 14, max_positions: int = 40) -> None:
        self._sma_window = sma_window
        self._atr_window = atr_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._sma_window).sort("session_date").tail(self._sma_window)
        if closes.height < self._sma_window + 1:
            return Signal(information_available_at=stamp, weights={})

        history = view.history()
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        sma_values = {symbol: (closes[symbol][-20:] / closes[symbol].shift(1)[-20:].mean()).iloc[-1] for symbol in symbols}
        atr_values = {symbol: _atr(history, symbol, self._atr_window) for symbol in symbols}

        ranked_symbols = sorted(symbols, key=lambda s: sma_values[s] - atr_values[s], reverse=True)
        top_symbols = ranked_symbols[:self._max_positions]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _atr(history: pl.DataFrame, symbol: str, window: int) -> float:
    high = history.select(pl.col("high")[symbol].to_list())[-window:]
    low = history.select(pl.col("low")[symbol].to_list())[-window:]
    close = history.select(pl.col("adj_close")[symbol].to_list())[-window:]

    true_range = (high - low).max().alias("tr") + (close.shift(-1) - high).abs().alias("tr2") + (low - close.shift(-1)).abs().alias("tr3")
    atr = true_range.mean().item()
    return atr