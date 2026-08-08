from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Companies that outperform their peers in terms of price appreciation may have better business fundamentals or market positioning. "
        "This strategy seeks to identify such companies by comparing the relative strength of each stock against a broad market index over a fixed period."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or "NIFTY100" not in view.symbols:
            return Signal(information_available_at=stamp, weights={})

        nifty_history = history.select(["session_date", pl.col("NIFTY100").alias("nifty_close")])
        other_symbols = [symbol for symbol in view.symbols if symbol != "NIFTY100"]
        symbols_data = []

        for symbol in other_symbols:
            symbol_df = history.select(["session_date", f"{symbol}"])
            symbol_df.rename({f"{symbol}": "close"}, in_place=True)
            combined_df = nifty_history.join(symbol_df, on="session_date")
            ratio_series = (combined_df["close"] / combined_df["nifty_close"]) - 1.0
            combined_df = combined_df.with_columns(ratio_series.alias(f"ratio_{symbol}"))

        combined_df = combined_df.sort("session_date", descending=False).tail(self._window)
        relative_strengths = combined_df.select(pl.all().mean())

        if relative_strengths.height == 0:
            return Signal(information_available_at=stamp, weights={})

        symbol_strengths = {
            symbol: float(relative_strengths[f"ratio_{symbol}"].item())
            for symbol in other_symbols
        }
        sorted_symbols = sorted(symbol_strengths.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_symbols[:5]]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest