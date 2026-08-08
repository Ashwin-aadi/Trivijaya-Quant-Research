from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the market in the recent past based on their "
        "price strength relative to the NIFTY 100 index can help identify potentially undervalued "
        "or well-performing stocks."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes(lookback=self._window).select(
            pl.col("^NSEI").alias("nifty_close")
        )
        stock_strength: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or "^NSEI" not in nifty_closes.columns:
                continue
            stock_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            nifty_values = [
                float(v) for v in nifty_closes["nifty_close"].drop_nulls().to_list()
            ]
            if len(stock_values) < self._window or len(nifty_values) < self._window:
                continue
            stock_returns = [
                (stock_values[i + 1] - stock_values[i]) / stock_values[i]
                for i in range(len(stock_values) - 1)
            ]
            nifty_returns = [
                (nifty_values[i + 1] - nifty_values[i]) / nifty_values[i]
                for i in range(len(nifty_values) - 1)
            ]
            if len(stock_returns) < self._window or len(nifty_returns) < self._window:
                continue
            stock_mean_return = sum(stock_returns[-self._window:]) / self._window
            nifty_mean_return = sum(nifty_returns[-self._window:]) / self._window
            strength_ratio = (stock_mean_return - nifty_mean_return) / nifty_mean_return if nifty_mean_return != 0 else 0.0
            stock_strength.append(strength_ratio)

        top_n_strength = sorted(stock_strength, reverse=True)[: self._top_n]
        picks: list[str] = []
        for symbol in view.symbols:
            strength_ratio = stock_strength[view.symbols.index(symbol)]
            if strength_ratio > max(top_n_strength) * 0.9 and strength_ratio >= 0.0:
                picks.append(symbol)

        picks = list(set(picks))
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