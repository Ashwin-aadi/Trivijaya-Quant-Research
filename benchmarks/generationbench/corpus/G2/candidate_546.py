from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength (RS) compared to the broader market "
        "tend to outperform in the long run. RS is calculated by comparing a stock's "
        "price movement over a period to that of the market index, NIFTY 100 in this case."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = [float(v) for v in history["adj_close"].to_list()]
        nifty100_returns = [(nifty100_closes[i] / nifty100_closes[i - 1] - 1.0) for i in range(1, self._window)]
        max_return = max(nifty100_returns)

        symbols_with_high_rs: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            returns = [(prices[i] / prices[i - 1] - 1.0) for i in range(1, self._window)]
            rs = sum([returns[i] * nifty100_returns[i] for i in range(self._window)]) / max_return
            if rs > 0.5:  # Threshold can be adjusted as needed
                symbols_with_high_rs.append(symbol)

        weight = 1.0 / len(symbols_with_high_rs)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_high_rs}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest