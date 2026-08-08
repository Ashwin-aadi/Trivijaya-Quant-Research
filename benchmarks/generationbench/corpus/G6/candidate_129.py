from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy selects stocks with lower historical volatility compared to the broader NIFTY 100 index. "
        "By focusing on low-volatility stocks, it aims to achieve a well-diversified portfolio while minimizing risk."
    )

    def __init__(self, window: int = 21, threshold_multiplier: float = 1.5) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history["symbol"].unique()) < 50:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        nifty_history = history.filter(pl.col("symbol").is_in(symbols))
        nifty_log_returns = _calculate_log_returns(nifty_history)
        nifty_volatility = nifty_log_returns.std()

        stock_log_returns = _calculate_log_returns(history, symbols=symbols)
        stock_volatilities = (stock_log_returns.std(axis=0) / nifty_volatility).to_list()
        ranked_symbols = [s for _, s in sorted(zip(stock_volatilities, symbols))]

        top_50_percent = int(len(ranked_symbols) * 0.5)
        selected_symbols = ranked_symbols[:top_50_percent]
        weights = {symbol: 0.02 for symbol in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_log_returns(history: pl.DataFrame, symbols: list[str] = None) -> pl.DataFrame:
    if symbols is not None:
        history = history.filter(pl.col("symbol").is_in(symbols))

    log_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1).alias("log_return")
    return history.with_columns(log_returns).sort("session_date")