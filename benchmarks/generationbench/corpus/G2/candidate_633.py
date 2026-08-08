from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically provided excess returns. The mechanism is "
        "believed to be related to risk compensation and investor behavior, where investors are "
        "willing to accept lower returns in exchange for reduced volatility."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        volatilities: dict[str, float] = {}

        for symbol in symbols:
            close_prices = [float(v) for v in history.select(pl.col("symbol") == symbol)[
                "adj_close"].to_list()]
            if len(close_prices) < self._lookback:
                continue

            returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                       for i in range(1, len(close_prices))]
            volatility = (sum([r ** 2 for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [s[0] for s in sorted(volatilities.items(), key=lambda item: item[1])]
        top_symbols = sorted_symbols[:min(len(sorted_symbols), 10)]
        weight = 1.0 / len(top_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest