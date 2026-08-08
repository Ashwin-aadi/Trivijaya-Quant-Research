from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top N symbols with the highest relative strength against "
        "the broad market index. Symbols with higher relative strength are expected to outperform."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength
        rel_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol == "NIFTY100":
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            nifty_close_values = [float(v) for v in closes["NIFTY100"].drop_nulls().to_list()]

            if len(close_values) < self._window or len(nifty_close_values) < self._window:
                continue

            rel_strength = (
                pl.DataFrame({"close": close_values, "nifty_close": nifty_close_values})
                .with_columns(
                    (pl.col("close") / pl.col("nifty_close").shift(1) - 1.0).alias("rs")
                )
                .select(pl.col("rs").mean())
                .item()
            )
            rel_strengths[symbol] = rel_strength

        # Sort symbols by relative strength and select top N
        sorted_symbols = sorted(rel_strengths.items(), key=lambda item: -item[1])
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]

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