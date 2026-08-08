from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Relative strength (RS) is a comparative measure of the strength of an asset compared to "
        "the overall market. Assets that are outperforming their peers relative to the broader "
        "market may have an increased likelihood of continued upward momentum."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_close = float(view.latest_close()[view.as_of])
        symbol_strengths: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            market_returns = [v / (values[i - 1] or 1.0) - 1.0 for i, v in enumerate(values)]
            symbol_mean_return = sum(market_returns[-self._window:]) / min(self._window, len(market_returns))
            strength = float(symbol_mean_return) / market_close
            if not pl.is_nan(strength):
                symbol_strengths[symbol] = strength

        top_symbols = sorted(symbol_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {symbol: value for symbol, value in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if not pl.is_nan(v)}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest