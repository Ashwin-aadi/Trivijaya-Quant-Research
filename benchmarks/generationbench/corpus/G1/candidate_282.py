from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Relative strength (RS) identifies the strength of a stock compared to its peers. "
        "By holding stocks that are outperforming their sector, we aim to capture excess returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        rs_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = closes[symbol].max().item()
            avg_price = (closes[symbol]
                         .sort("session_date", descending=True)
                         .head(self._window)
                         .mean()
                         .item())
            rs_score = latest_close / avg_price - 1.0
            rs_scores[symbol] = rs_score

        sorted_rs = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_rs[:self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).item()
    assert isinstance(newest, date)
    return newest