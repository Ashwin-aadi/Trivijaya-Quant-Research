from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsSurpriseLiquidity(Strategy):
    rationale = (
        "This strategy identifies stocks with significant positive earnings surprises combined "
        "with low market liquidity. By focusing on such characteristics, the aim is to capitalize "
        "on potential mispricings due to insufficient trading activity and strong earnings "
        "performance."
    )

    def __init__(self, earnings_window: int = 30, liquidity_window: int = 90, top_n: int = 10) -> None:
        self._earnings_window = earnings_window
        self._liquidity_window = liquidity_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._earnings_window + self._liquidity_window)
        if history.height < self._earnings_window + self._liquidity_window:
            return Signal(information_available_at=stamp, weights={})

        # Extract relevant columns
        symbols = view.symbols
        closes = view.closes()

        # Calculate earnings surprise
        q1_close = history.select(pl.col("adj_close").filter(pl.col("session_date") == date(2024, 3, 31))).select(
            pl.col(symbols)
        )
        q1_close = {symbol: float(close) for symbol, close in zip(symbols, q1_close.to_numpy().flatten())}
        
        consensus_estimates = history.select(pl.col("adj_close").filter(pl.col("session_date") == date(2024, 3, 30))).select(
            pl.col(symbols)
        )
        consensus_estimates = {symbol: float(estimate) for symbol, estimate in zip(symbols, consensus_estimates.to_numpy().flatten())}
        
        earnings_surprise = {
            symbol: (q1_close[symbol] - consensus_estimates[symbol]) / consensus_estimates[symbol]
            for symbol in symbols
        }
        
        # Filter stocks with at least 10% earnings surprise and market cap < $5B
        eligible_symbols = [
            symbol for symbol, surprise in earnings_surprise.items() if surprise >= 0.1 and view.latest_close()[symbol] < 5_000_000_000
        ]
        
        # Calculate liquidity measure (avg daily trading volume)
        avg_vol = history.select(pl.col("volume")).group_by("symbol").agg(
            pl.col("volume").mean().alias("avg_vol")
        )
        avg_vol = {symbol: float(avg_vol.filter(pl.col("symbol") == symbol)["avg_vol"].item()) for symbol in symbols}
        
        # Filter out high-liquidity stocks
        low_liquid_symbols = [symbol for symbol, vol in avg_vol.items() if vol <= 2_000_000]
        
        # Intersection of eligible and low-liquid stocks
        final_symbols = set(eligible_symbols) & set(low_liquid_symbols)
        
        # Rank based on earnings surprise (descending order)
        ranked_symbols = sorted(final_symbols, key=lambda x: earnings_surprise[x], reverse=True)[: self._top_n]
        
        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest