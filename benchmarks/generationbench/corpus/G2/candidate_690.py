from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with strong relative performance tend to continue outperforming the broader market. "
        "This strategy aims to identify and invest in the top-performing stocks over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative strength by comparing each stock's performance to the market
        closes = view.closes(lookback=self._window)
        market_close = float(view.latest_close()["^NIFTY 100"])
        strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol == "^NIFTY 100":
                continue
            if symbol not in closes.columns:
                continue
            values = [float(v) / market_close for v in closes[symbol].to_list()]
            if len(values) < self._window:
                continue
            avg_strength = sum(values) / self._window
            strengths[symbol] = avg_strength

        # Sort symbols by their relative strength and pick the top N
        sorted_stocks = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
        picks = [stock for stock, _ in sorted_stocks[:5]]
        
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