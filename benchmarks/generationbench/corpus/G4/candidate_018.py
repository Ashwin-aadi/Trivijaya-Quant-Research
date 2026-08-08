from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy exploits the tendency of outperforming stocks to continue outperforming "
        "by selecting stocks with a higher relative strength index (RSI) compared to the Nifty 50 or S&P BSE Sensex over the past 6 months. It aims to capture gains from outperforming stocks while managing risk through periodic adjustments and diversification."
    )

    def __init__(self, window: int = 180, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50 = view.closes(lookback=self._window).to_series().rename("NIFTY50")
        symbols = view.symbols

        def rsi_calculation(df: pl.DataFrame) -> float:
            close = df["close"]
            delta = close.diff().drop_nulls()
            gain = (delta.where(delta > 0, other=0)).sum() / self._window
            loss = (-delta.where(delta < 0, other=0)).sum() / self._window
            rs = gain / loss if loss != 0 else float("inf")
            return 100 - (100 / (1 + rs))

        rsi_scores: dict[str, float] = {}
        for symbol in symbols:
            close_series = history.select(pl.col("close").filter(pl.col("symbol") == symbol)).to_series().rename(symbol)
            combined = pl.concat([nifty50, close_series]).sort(by="session_date")
            rsi_score = rsi_calculation(combined.slice(self._window))
            rsi_scores[symbol] = rsi_score

        sorted_symbols = [k for k, v in sorted(rsi_scores.items(), key=lambda item: -item[1])][: self._top_n]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().to_list()[0]
    assert isinstance(newest, date)
    return newest