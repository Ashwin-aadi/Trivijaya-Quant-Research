from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks that have outperformed the NIFTY 100 index in the recent past. "
        "Outperforming stocks are expected to continue their upward trend and offer better returns."
    )

    def __init__(self, window: int = 20, threshold: float = 1.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = [float(v) for v in view.closes().select("NIFTY 100").to_list()[0]]
        others_closes = {symbol: [float(v) for v in history.select(symbol).to_columns()[0]] for symbol in view.symbols if symbol != "NIFTY 100"}

        outperforming_symbols = []
        for symbol, closes in others_closes.items():
            if len(closes) < self._window:
                continue
            returns = [c / nifty_closes[i - 1] - 1.0 for i, c in enumerate(closes)]
            if max(returns) > self._threshold:
                outperforming_symbols.append(symbol)

        weight = 1.0 / len(outperforming_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in outperforming_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest