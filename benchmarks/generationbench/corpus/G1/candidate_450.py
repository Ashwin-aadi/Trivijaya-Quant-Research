from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the broader market can help in capturing "
        "momentum and potentially higher returns. This strategy ranks assets based on their "
        "performance relative to the NIFTY 100 index and allocates capital to the top performers."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_close = history.select(pl.col("adj_close").filter(pl.col("symbol") == "NIFTY")).to_series().item()
        symbol_returns: dict[str, float] = {}
        
        for symbol in view.symbols:
            if symbol != "NIFTY":
                close_prices = [float(v) for v in history.select(symbol).to_series().drop_nulls().to_list()]
                if len(close_prices) >= self._lookback:
                    returns = [(close_prices[i] / close_prices[i - 1]) - 1.0 for i in range(1, self._lookback)]
                    avg_return = sum(returns) / len(returns)
                    symbol_returns[symbol] = avg_return

        sorted_symbols = [k for k, v in sorted(symbol_returns.items(), key=lambda item: item[1], reverse=True)[:5]]
        
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).to_pydate()
    assert isinstance(newest, date)
    return newest