from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the broader NIFTY 100 index "
        "tends to outperform over time. This strategy identifies and invests in the top-performing "
        "stocks relative to the NIFTY 100."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes().select("NIFTY 100").rename({"NIFTY 100": "nifty"})
        non_nifty_symbols = [s for s in view.symbols if s != "NIFTY 100"]
        symbol_history = history.select(non_nifty_symbols).with_columns(
            (pl.col(symbol) / pl.col("NIFTY 100") - 1.0).alias(f"rel_strength_{symbol}")
            for symbol in non_nifty_symbols
        )

        rel_strength_scores = symbol_history.select(
            *[
                pl.col(f"rel_strength_{s}").mean().alias(s)
                for s in non_nifty_symbols
            ]
        ).to_numpy()

        if rel_strength_scores.size == 0:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [non_nifty_symbols[i] for i in rel_strength_scores.argsort()[::-1][:3]]
        weight = 1.0 / len(top_symbols) if top_symbols else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest