from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the liquidity premium by equally weighting stocks that meet "
        "certain liquidity criteria. Higher trading volumes and fewer price impacts lead to better"
        " information dissemination and outperformance in the Indian equity market."
    )

    def __init__(self, turnover_threshold: float = 0.01, top_n: int = 50) -> None:
        self._turnover_threshold = turnover_threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        data = history.to_pandas()
        symbols = data["symbol"].tolist()

        def daily_turnover(open_price: float, high_price: float, low_price: float,
                           close_price: float, volume: int, market_cap: float) -> float:
            trading_volume = (high_price + low_price + open_price + close_price) / 4 * volume
            return trading_volume / market_cap

        turnover_ratios = [
            daily_turnover(
                data.loc[data["symbol"] == symbol, "open"].iloc[0],
                data.loc[data["symbol"] == symbol, "high"].iloc[0],
                data.loc[data["symbol"] == symbol, "low"].iloc[0],
                data.loc[data["symbol"] == symbol, "close"].iloc[0],
                data.loc[data["symbol"] == symbol, "volume"].iloc[0],
                view.latest_close()[symbol] * 1000  # Assume market cap is in lakhs
            )
            for symbol in symbols
        ]

        filtered_symbols = [s for s, ratio in zip(symbols, turnover_ratios) if ratio >= self._turnover_threshold]
        selected_symbols = filtered_symbols[:self._top_n]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest