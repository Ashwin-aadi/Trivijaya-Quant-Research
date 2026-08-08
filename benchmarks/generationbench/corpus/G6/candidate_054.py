from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength compared to the NIFTY 50 index. "
        "Stocks are added if they outperform the index by more than 3% over a 21-day period and removed "
        "if they underperform by -1% or after holding for 6 trading days."
    )

    def __init__(self, window: int = 21, threshold_gain: float = 0.03, threshold_loss: float = -0.01, max_holding_days: int = 6) -> None:
        self._window = window
        self._threshold_gain = threshold_gain
        self._threshold_loss = threshold_loss
        self._max_holding_days = max_holding_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        nifty50_history = view.history(lookback=self._window + 1).filter(pl.col("symbol") == "NIFTY50")
        if nifty50_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = nifty50_history.select("adj_close").collect().to_numpy()[0].tolist()
        nifty50_returns = [(nifty50_closes[i] - nifty50_closes[i-1]) / nifty50_closes[i-1] for i in range(1, len(nifty50_closes))]

        stock_history = view.history(lookback=self._window)
        relevant_stocks = [symbol for symbol in view.symbols if symbol != "NIFTY50" and symbol in stock_history.columns]
        
        relative_strength: dict[str, float] = {}
        for symbol in relevant_stocks:
            close_prices = [float(v) for v in stock_history[symbol].select("adj_close").to_numpy().tolist()[1]]
            returns = [(close_prices[i] - close_prices[i-1]) / close_prices[i-1] for i in range(1, len(close_prices))]
            
            if sum(returns) / self._window > nifty50_returns[-1] * (1 + self._threshold_gain):
                relative_strength[symbol] = 1.0

        picks = [s for s, strength in relative_strength.items() if strength == 1.0]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        signal = Signal(
            information_available_at=stamp, 
            weights={p: weight for p in picks}
        )

        # Check and remove stocks that have been held for more than max_holding_days
        current_portfolio = {s: view.latest_close()[s] for s in signal.weights.keys()}
        to_remove = [s for s, price in current_portfolio.items() if (stamp - view.as_of).days > self._max_holding_days and relative_strength[s] != 1.0]
        
        if to_remove:
            for stock_to_remove in to_remove:
                del signal.weights[stock_to_remove]

        return signal


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest