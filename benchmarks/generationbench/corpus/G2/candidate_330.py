from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed relative to the broad market over a short period are "
        "likely to continue to do so due to persistence in asset returns. This strategy "
        "identifies such assets and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_avg_returns = _calculate_market_average_return(closes)
        asset_returns = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, self._window + 1)]
            avg_return = sum(returns) / len(returns)
            asset_returns[symbol] = avg_return - market_avg_returns

        top_assets = sorted(asset_returns.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {symbol: weight for symbol, weight in top_assets}

        if not weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={s: w / sum(weights.values()) for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_market_average_return(closes: pl.DataFrame) -> float:
    market_close = closes.select(pl.col("adj_close").mean().alias("avg")).collect().rows()[0][0]
    returns = []
    for i in range(1, closes.height):
        returns.append((market_close - closes["adj_close"][i]) / closes["adj_close"][i-1])
    
    return sum(returns) / len(returns)