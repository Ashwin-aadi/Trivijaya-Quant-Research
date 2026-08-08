from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets in an uptrend are "
        "more likely to continue trending upward. By scaling our position size based on the "
        "recent volatility of each asset, we can capture more returns from strong trends while "
        "limiting exposure during periods of low volatility."
    )

    def __init__(self, window: int = 20, trend_factor: float = 1.5) -> None:
        self._window = window
        self._trend_factor = trend_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
            trend_score = self._calculate_trend_score(prices)
            vol = pl.DataFrame({"close": prices}).select(
                (pl.col("close").rolling_std(window=self._window)).alias("vol")
            ).collect()["vol"][0]
            symbol_data[symbol] = {"trend_score": trend_score, "volatility": vol}

        sorted_symbols = sorted(symbol_data.items(), key=lambda x: -x[1]["trend_score"])
        top_symbols = [symbol for symbol, data in sorted_symbols[: self._top_n]]
        
        weights = {}
        total_volatility = sum([symbol_data[symbol]["volatility"] for symbol in top_symbols])
        for symbol in top_symbols:
            volatility = symbol_data[symbol]["volatility"]
            weight = (volatility / total_volatility) * self._trend_factor
            weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_trend_score(prices: list[float]) -> float:
    returns = [p / prices[i - 1] - 1 for i, p in enumerate(prices[1:], start=1)]
    trend_score = (sum(returns) / len(returns)) * len(returns)
    return trend_score