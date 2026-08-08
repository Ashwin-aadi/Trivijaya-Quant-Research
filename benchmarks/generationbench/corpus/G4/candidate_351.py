from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the anomaly of smaller-cap stocks by screening based on liquidity and applying equal weighting. "
        "It aims to capture higher returns from smaller caps while mitigating risks through balanced exposure."
    )

    def __init__(self, window_liquid: int = 30, window_rank: int = 10, n_stocks: int = 25) -> None:
        self._window_liquid = window_liquid
        self._window_rank = window_rank
        self._n_stocks = n_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_liquid + self._window_rank)
        if history.height < self._window_liquid + self._window_rank:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily turnover ratio for liquidity screening
        volume_history = history.select(
            pl.col("symbol"),
            (pl.col("volume").sum() / pl.sum("adj_close") * 2).alias("turnover_ratio")
        )
        
        # Screen out illiquid stocks based on turnover ratio
        liquid_symbols = [sym for sym, tr in zip(volume_history["symbol"], volume_history["turnover_ratio"].to_list()) if tr > 0.5]
        
        # Filter history to only include liquid symbols
        filtered_history = history.filter(pl.col("symbol").is_in(liquid_symbols))

        # Select smaller-cap stocks based on market capitalization (example threshold)
        small_caps = [sym for sym, mkt_cap in zip(view.symbols, view.closes().select(pl.col("symbol"), "adj_close").to_dict(False).values()) if mkt_cap < 500]

        # Further filter to include only small-cap liquid stocks
        candidates = set(liquid_symbols) & set(small_caps)

        # Rank the remaining candidates based on recent performance (simple example: last close price)
        ranked_candidates = [sym for sym, close in zip(candidates, filtered_history.filter(pl.col("symbol").is_in(candidates))["adj_close"].to_list()) if len(close) >= self._window_rank]
        
        picks = sorted(ranked_candidates, key=lambda x: filtered_history.filter(pl.col("symbol") == x)["adj_close"].tail(self._window_rank).sum(), reverse=True)[:self._n_stocks]

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
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest