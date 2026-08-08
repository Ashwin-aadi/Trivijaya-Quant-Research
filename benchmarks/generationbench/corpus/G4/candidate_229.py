from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion520(Strategy):
    rationale = (
        "This strategy capitalizes on short-term deviations from stock prices' mean levels in the Indian equity market. By identifying stocks that have significantly deviated from their moving average within a 5-20 day window, it seeks to profit from mean reversion tendencies."
    )

    def __init__(self, window_min: int = 5, window_max: int = 20, top_n: int = 30) -> None:
        self._window_min = window_min
        self._window_max = window_max
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_max + 1).sort("session_date")
        if history.height < self._window_min:
            return Signal(information_available_at=stamp, weights={})

        smas = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "adj_close" not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].filter(pl.col("session_date") > (stamp - date(self._window_max))).drop_nulls().to_list()]
            sma = sum(adj_closes[-self._window_min:]) / self._window_min
            smas[symbol] = sma

        buys, sells = [], []
        for symbol in view.symbols:
            if symbol not in smas or symbol not in history.columns:
                continue
            adj_close = float(history[symbol].filter(pl.col("session_date") == stamp).to_list()[0][1])
            diff = abs(adj_close - smas[symbol])

            if adj_close < smas[symbol]:
                buys.append((symbol, diff))
            elif adj_close > smas[symbol]:
                sells.append((symbol, diff))

        buys.sort(key=lambda x: x[1])
        sells.sort(key=lambda x: x[1], reverse=True)

        longs = [b[0] for b in buys[: self._top_n]]
        shorts = [s[0] for s in sells[: self._top_n]]

        if not (longs or shorts):
            return Signal(information_available_at=stamp, weights={})

        weight_long = 2.0 / len(longs) if longs else 0
        weight_short = -2.0 / len(shorts) if shorts else 0

        weights = {symbol: max(min(weight_long, 1), -weight_short) for symbol in view.symbols}

        return Signal(information_available_at=stamp, weights={**weights})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest