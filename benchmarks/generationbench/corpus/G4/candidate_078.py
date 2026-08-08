from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIOutperformer(Strategy):
    rationale = (
        "This strategy identifies stocks with a higher Relative Strength Index (RSI) relative "
        "to the NIFTY 50 index. Stocks that outperform the broader market by having an RSI above "
        "70 are selected for inclusion, capturing momentum and persistent performance."
    )

    def __init__(self, window: int = 90, threshold: float = 70, top_n: int = 20) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_history = view.closes(lookback=self._window).select("NIFTY 50")
        if nifty50_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        stock_rsis = _compute_relative_strength_index(history, "NIFTY 50", window=self._window)
        strong_stocks = [s for s, rsi in stock_rsis.items() if rsi > self._threshold]
        top_n_strong_stocks = strong_stocks[: self._top_n]

        if not top_n_strong_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_strong_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_strong_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_relative_strength_index(history: pl.DataFrame, index_symbol: str, window: int) -> dict[str, float]:
    index_data = history.select(index_symbol).rename({"NIFTY 50": "index"})
    stock_data = history.select([pl.col(c).alias(f"{c}_adj_close") for c in view.symbols])

    closes = pl.concat([index_data, stock_data], how="horizontal").sort("session_date")
    closes["index_pct_change"] = (
        (closes[f"index_adj_close"] / closes[f"index_adj_close"].shift(1) - 1.0).fill_null(0)
    )
    for symbol in view.symbols:
        closes[f"{symbol}_pct_change"] = (
            (closes[f"{symbol}_adj_close"] / closes[f"{symbol}_adj_close"].shift(1) - 1.0).fill_null(0)
        )

    rsi_values = _calculate_rsi(closes, window)
    return {s: float(rsi_values[s]) for s in view.symbols if s in rsi_values}


def _calculate_rsi(data: pl.DataFrame, window: int) -> dict[str, float]:
    def compute_rsi(expr):
        return (
            (expr.rolling_sum(window).mean() / expr.rolling_mean(window)).alias(f"rsi_{window}")
        )

    data = data.sort("session_date")
    rsi_values = {}
    for symbol in view.symbols:
        rsi_expr = _compute_relative_strength(data[f"{symbol}_pct_change"], window)
        rsi_values[symbol] = float(rsi_expr.with_columns(compute_rsi(rsi_expr)).select(f"rsi_{window}").row(0)[0])
    return rsi_values


def _compute_relative_strength(changes: pl.DataFrame, window: int) -> pl.Expr:
    gain = (changes.where(changes > 0).rolling_mean(window)).alias("gain")
    loss = (changes.where(changes < 0).abs().rolling_mean(window)).alias("loss")
    rs = gain / loss
    return Expr.rank(method="dense", descending=True).alias(f"rs_{window}")