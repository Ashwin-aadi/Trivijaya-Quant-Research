from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks in India that outperform the S&P BSE Sensex over a 6-month period. "
        "By focusing on relative strength, it aims to capture the upside potential of outperforming stocks while minimizing exposure to underperformers."
    )

    def __init__(self, lookback_period: int = 180, top_n: int = 4) -> None:
        self._lookback_period = lookback_period
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.height < self._lookback_period + 1:
            return Signal(information_available_at=stamp, weights={})

        sensex_symbols = ("^BSESN",)  # Assuming S&P BSE Sensex as the benchmark
        non_sensex_symbols = [symbol for symbol in view.symbols if symbol not in sensex_symbols]

        sensex_history = history.filter(pl.col("symbol").is_in(sensex_symbols))
        stock_histories = [history.filter(pl.col("symbol") == sym) for sym in non_sensex_symbols]

        sensex_returns = _compute_returns(sensex_history)
        stock_returns = [_compute_returns(hist) for hist in stock_histories]

        relative_strength_scores = _rank_relative_strength(stock_returns, sensex_returns)

        top_stocks = [stock for score, stock in sorted(relative_strength_scores.items(), key=lambda x: -x[0])[: self._top_n]]
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={stock: weight for stock in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_returns(history: pl.DataFrame) -> list[float]:
    latest_close = history.select(pl.col("close").last()).to_list()[0][0]
    returns = []
    for close in history["adj_close"].drop_nulls().to_list():
        if close is not None:
            returns.append(float(close / latest_close - 1))
    return returns


def _rank_relative_strength(stock_returns: list[list[float]], sensex_returns: list[float]) -> dict[float, str]:
    stock_scores = {}
    for i, stock_return in enumerate(stock_returns):
        stock_mean_return = sum(stock_return) / len(stock_return)
        sensex_mean_return = sum(sensex_returns) / len(sensex_returns)
        relative_strength_score = (stock_mean_return - sensex_mean_return) / max([abs(stock_mean_return), abs(sensex_mean_return)], default=1)
        stock_scores[relative_strength_score] = non_sensex_symbols[i]
    return stock_scores