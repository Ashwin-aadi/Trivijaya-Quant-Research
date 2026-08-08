from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIOutperform(Strategy):
    rationale = (
        "By selecting stocks with a higher Relative Strength Index (RSI) compared to the broader "
        "market index like NIFTY 50, this strategy aims to capture alpha from overbought conditions in high-quality stocks."
    )

    def __init__(self, window: int = 14, top_n_percent: float = 0.2) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_symbols = ("NIFTY 50").split()
        universe_symbols = view.symbols

        # Compute RSI for NIFTY 50
        nifty_history = history.filter(pl.col("symbol").is_in(nifty_symbols))
        rsi_nifty = _compute_rsi(nifty_history, self._window)

        # Compute RSI for all symbols in the universe
        universe_history = history.filter(pl.col("symbol").is_in(universe_symbols))
        rsi_universe = _compute_rsi(universe_history, self._window)

        # Rank stocks by their RSI relative to NIFTY 50
        rsi_ratio = (rsi_universe["adj_close"] / rsi_nifty["adj_close"]).to_list()
        sorted_indices = [universe_symbols[i] for i in _rank_series(rsi_ratio, method="dense", descending=True)]
        
        top_n = int(len(sorted_indices) * self._top_n_percent)
        top_symbols = sorted_indices[:top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate portfolio weights based on RSI values
        rsi_values_universe = [float(rsi_universe[symbol].to_list()[-1]) for symbol in top_symbols]
        total_rsi = sum(rsi_values_universe)
        weights = {symbol: value / total_rsi for symbol, value in zip(top_symbols, rsi_values_universe)}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(history: pl.DataFrame, window: int) -> pl.DataFrame:
    def rsi(values):
        delta = values.diff().shift(-1).fillna(0)
        up = delta.where(delta > 0).mean()
        down = (-delta.where(delta < 0)).mean()
        rs = up / (down + 1e-8)  # Avoid division by zero
        return 100 - 100 / (1 + rs)

    rsi_values = history.groupby("symbol").agg(
        pl.col("adj_close")
        .rolling_mean(window)
        .alias(f"mean_{window}")
    )
    rsi_values = rsi_values.with_columns(
        (rsi(rsi_values[f"adj_close"], window)).alias(f"RSI_{window}")
    ).sort("symbol", descending=True)

    return rsi_values


def _rank_series(series: list[float], method: str, descending: bool) -> pl.Series:
    rank = pl.Series(series).rank(method=method, descending=descending)
    return rank.to_list()