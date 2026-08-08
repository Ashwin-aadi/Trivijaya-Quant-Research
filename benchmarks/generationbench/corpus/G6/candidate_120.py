from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum and fundamental valuation by selecting "
        "stocks with strong recent performance and low valuations. It aims to capture gains from "
        "momentum stocks while mitigating risk through strict exit rules."
    )

    def __init__(self, window: int = 60, top_n_percent: float = 0.3, valuation_window: int = 252) -> None:
        self._window = window
        self._top_n_percent = top_n_percent
        self._valuation_window = valuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        valuation_history = view.history(lookback=self._valuation_window)

        # Calculate cumulative returns
        returns = (closes["close"] / closes["close"].shift(self._window) - 1.0).alias("cumulative_return")
        history = history.with_columns(returns)
        
        # Calculate volume growth
        volume_growth = ((closes["volume"] / closes["volume"].shift(1)) - 1.0).alias("volume_growth")
        history = history.with_columns(volume_growth)

        # Rank by cumulative return and volume growth
        ranked_history = (
            history.group_by("symbol").agg(
                pl.col("cumulative_return").mean().alias("avg_cumulative_return"),
                pl.col("volume_growth").mean().alias("avg_volume_growth")
            )
        ).sort("avg_cumulative_return", descending=True).sort("avg_volume_growth", descending=True)

        # Get top N% of symbols
        num_symbols = len(view.symbols)
        top_n_count = int(num_symbols * self._top_n_percent)
        picks = ranked_history.head(top_n_count)["symbol"].to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Calculate valuation metrics (P/E and P/B below market median)
        market_median_pe = float(valuation_history.select(pl.col("pe_ratio").median())["pe_ratio"][0])
        market_median_pb = float(valuation_history.select(pl.col("pb_ratio").median())["pb_ratio"][0])

        picks_valuation_filtered: list[str] = []
        for symbol in picks:
            pe = float(view.history(lookback=self._valuation_window)[symbol]["pe_ratio"].max())
            pb = float(view.history(lookback=self._valuation_window)[symbol]["pb_ratio"].max())
            if pe < market_median_pe and pb < market_median_pb:
                picks_valuation_filtered.append(symbol)

        if not picks_valuation_filtered:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks_valuation_filtered)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks_valuation_filtered}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest