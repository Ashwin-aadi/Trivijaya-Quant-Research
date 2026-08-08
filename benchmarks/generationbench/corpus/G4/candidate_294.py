from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength strategy exploits the tendency of outperforming assets to continue "
        "outperforming due to factors like positive news, investor sentiment, or competitive advantages. "
        "By focusing on stocks with strong relative performance against a broad universe index (NIFTY 50), "
        "we aim to benefit from both intra-market momentum and potentially underpriced opportunities in outperforming sectors or companies."
    )

    def __init__(self, window: int = 120, top_n_percent: float = 0.3) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50 = NIFTY50().history(lookback=self._window)
        if nifty50.is_empty() or nifty50.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = nifty50["adj_close"]
        symbols = view.symbols

        # Compute cumulative returns for NIFTY 50
        nifty50_returns = (nifty50_closes / nifty50_closes.shift(1) - 1.0).sum()

        # Compute cumulative returns for each stock
        stock_returns = {}
        for symbol in symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(closes) < self._window:
                continue

            # Calculate cumulative return for the stock
            stock_returns[symbol] = (pl.Series(closes) / pl.Series(closes).shift(1) - 1.0).sum()

        # Compute Relative Strength scores
        rs_scores = {symbol: stock_returns[symbol] / nifty50_returns for symbol in symbols}

        # Sort stocks by RS score and select top N%
        sorted_stocks = sorted(rs_scores.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(symbols) * self._top_n_percent)

        picks = [symb for symb, _ in sorted_stocks[:top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def NIFTY50() -> pl.DataFrame:
    # Dummy function to represent the NIFTY50 index data
    df = (
        view.history(lookback=None)
        .filter(pl.col("symbol") == "NIFTY 50")
        .select(["session_date", "adj_close"])
    )
    return df


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest