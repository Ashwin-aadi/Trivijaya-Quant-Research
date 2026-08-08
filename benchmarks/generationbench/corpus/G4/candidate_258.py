from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "By selecting stocks based on their relative strength against the broad market universe, "
        "we aim to capitalize on persistent performance trends. Strong performers are expected to "
        "outperform weak ones over time due to underlying structural or fundamental advantages."
    )

    def __init__(self, lookback: int = 12, top_n: int = 10) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue
            last_close = values[-1]
            start_close = values[0]
            cumulative_return = (last_close / start_close - 1.0) * 100
            cumulative_returns[symbol] = cumulative_return

        ranked_symbols = sorted(cumulative_returns.items(), key=lambda x: x[1], reverse=True)
        top_decile, bottom_decile = _get_deciles(ranked_symbols)

        if not top_decile or not bottom_decile:
            return Signal(information_available_at=stamp, weights={})

        weight_top = 0.05 / len(top_decile)
        weight_bottom = -0.05 / len(bottom_decile)
        weights = {symbol: weight_top for symbol in top_decile}
        for symbol in bottom_decile:
            weights[symbol] = weight_bottom

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _get_deciles(ranked_symbols: list[tuple[str, float]]) -> tuple[list[str], list[str]]:
    total_stocks = len(ranked_symbols)
    top_n = min(total_stocks // 10 * 2, total_stocks)  # Top two deciles combined
    bottom_n = min(total_stocks - top_n, total_stocks // 10)

    return [symbol for symbol, _ in ranked_symbols[:top_n]], [symbol for symbol, _ in ranked_symbols[-bottom_n:]]