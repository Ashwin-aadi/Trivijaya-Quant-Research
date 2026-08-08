from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts followed by sustained price movements are often indicative of a shift in market "
        "trend. This strategy aims to capture these trends by identifying breakouts above or below "
        "significant support and resistance levels, ensuring the breakout is confirmed with increased "
        "volume and then entering at the breakout price with a trailing stop."
    )

    def __init__(self, window: int = 50, min_volume_multiplier: float = 1.5) -> None:
        self._window = window
        self._min_volume_multiplier = min_volume_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            hist = history.select(
                pl.col("symbol").eq(symbol).alias("match"),
                pl.col("close").shift(-1).alias("next_close"),
                (pl.col("volume") / pl.col("volume").mean()).alias("vol_ratio")
            ).filter(pl.col("match"))
            
            if hist.height == 0:
                continue
            
            close = float(hist["close"].to_list()[-1])
            next_close = float(hist["next_close"].to_list()[0])
            vol_ratio = float(hist["vol_ratio"].mean().round(2))
            
            if close < history.select(pl.col("symbol").eq(symbol)).sort("session_date", descending=True).tail(5)["close"].to_list()[-1]:
                continue
            
            if next_close > close and vol_ratio >= self._min_volume_multiplier:
                breakout_symbols.append(symbol)

        top_n = min(len(breakout_symbols), 3)
        picks = breakout_symbols[:top_n]
        
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