from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have recently broken out from their historical "
        "trading ranges and then monitors them for continuation patterns. By capitalizing on the "
        "momentum of breakouts, it aims to benefit from the reinforcement of price movements by "
        "investor behavior and institutional rebalancing strategies."
    )

    def __init__(self, window: int = 20, min_breakout_pct: float = 5.0, top_n: int = 10) -> None:
        self._window = window
        self._min_breakout_pct = min_breakout_pct
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            hist = view.history(lookback=self._window).filter(pl.col("symbol") == symbol)
            high = hist.select(pl.col("adj_close").max().alias("high"))[0]["high"]
            low = hist.select(pl.col("adj_close").min().alias("low"))[0]["low"]
            today_close = view.latest_close()[symbol]
            breakout_pct = (today_close - min(high, low)) / max(high, low) * 100

            if breakout_pct >= self._min_breakout_pct:
                next_day_close = closes[symbol].shift(-1).to_list()
                if next_day_close[-1] > today_close and today_close >= high:
                    breakout_signals.append(symbol)
                elif next_day_close[-1] < today_close and today_close <= low:
                    breakout_signals.append(symbol)

        breakout_signals = breakout_signals[: self._top_n]
        if not breakout_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest