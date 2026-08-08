from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where smaller-cap stocks often exhibit higher "
        "returns due to mispriced liquidity. By screening for companies with better liquidity "
        "metrics and then equal-weighting these in a portfolio, we can capture excess returns."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        turnover_ratios: list[float] = []
        trading_volumes: list[int] = []

        for symbol in view.symbols:
            history = view.history(symbol=symbol)
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue
            trading_volume = sum(history["volume"].to_list())
            market_cap = _market_cap(view.symbols, symbol)
            turnover_ratio = (trading_volume / max(1, market_cap)) * 100

            turnover_ratios.append(turnover_ratio)
            trading_volumes.append(trading_volume)

        liquidity_scores = [t / v for t, v in zip(turnover_ratios, trading_volumes)]
        ranked_symbols = [
            s
            for _, s in sorted(zip(liquidity_scores, view.symbols), reverse=True)[: self._top_n]
        ]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _market_cap(symbols: tuple[str, ...], symbol: str) -> float:
    close_prices = [float(v) for v in view.closes(lookback=20)[symbol].to_list()]
    if len(close_prices) < 20:
        return 1.0
    latest_price = close_prices[-1]
    shares_outstanding = _shares_outstanding(symbols, symbol)
    return latest_price * shares_outstanding


def _shares_outstanding(symbols: tuple[str, ...], symbol: str) -> float:
    # For simplicity, assume a constant share outstanding for all symbols
    return 1.0e6