from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves buying stocks that have outperformed the market "
        "in recent periods. The idea is to exploit persistent stock returns based on past performance."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback - 1)
        
        if not history.select("session_date").is_empty() and history.height >= self._window + self._lookback - 1:
            latest_closes = {symbol: float(close) for symbol, close in view.latest_close().items()}
            closes = (
                history
                .select(pl.col("session_date"), *[pl.col(symbol).alias(f"close_{symbol}") for symbol in view.symbols])
            )
            
            # Calculate daily returns
            returns = (
                closes.with_columns(
                    (pl.col(f"close_{symbol}") / pl.col(f"close_{symbol}").shift(1) - 1.0).alias(f"return_{symbol}")
                    for symbol in view.symbols
                )
            )

            recent_returns = (
                returns.filter(pl.col("session_date") >= pl.col("session_date").max().shift(-self._lookback + 1))
                .select([f"return_{symbol}" for symbol in view.symbols])
                .group_by("session_date")
                .agg(
                    [(pl.col(f"return_{symbol}").mean().alias(f"avg_return_{symbol}")) 
                     for symbol in view.symbols]
                )
            )

            # Find symbols with the highest mean returns
            top_symbols = recent_returns.sort([f"avg_return_{symbol}" for symbol in view.symbols], descending=True).head(self._lookback)
            
            picks: list[str] = [
                s for s in view.symbols 
                if f"avg_return_{s}" in top_symbols.columns and latest_closes[s] == float(top_symbols[f"avg_return_{s}"].first())
            ]
            
            if not picks:
                return Signal(information_available_at=stamp, weights={})

            weight = 1.0 / len(picks)
            return Signal(
                information_available_at=stamp, 
                weights={symbol: weight for symbol in picks}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest