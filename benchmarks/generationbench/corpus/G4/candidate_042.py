from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy exploits the relative strength of stocks against a broad market index. "
        "By investing in outperforming assets over a lookback period, we aim to capture alpha while "
        "mitigating risk through periodic rebalancing."
    )

    def __init__(self, window: int = 180, top_n_percent: float = 0.20, rebalance_period: int = 12) -> None:
        self._window = window
        self._top_n_percent = top_n_percent
        self._rebalance_period = rebalance_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        benchmark_closes = view.closes(lookback=self._window)["NIFTY 50"].drop_nulls().to_list()
        stock_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].select("adj_close").drop_nulls().to_list()]
            benchmark_price = float(benchmark_closes[-1])
            price6monthsago = float(prices[0])
            return_6_months = (benchmark_price - price6monthsago) / price6monthsago
            stock_returns[symbol] = return_6_months

        ranked_stocks = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)
        top_n_count = int(len(ranked_stocks) * self._top_n_percent)
        picks = [symbol for symbol, _ in ranked_stocks[:top_n_count]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.5 / len(picks)
        signal_weights = {s: weight for s in picks}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest