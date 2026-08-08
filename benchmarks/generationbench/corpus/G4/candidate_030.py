from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy exploits the tendency for stocks with strong relative performance "
        "against a broad index (such as NIFTY 50) to continue outperforming over time. By focusing on "
        "relative strength, we can capture gains from both stock-specific factors and market-wide momentum effects."
    )

    def __init__(self, window: int = 252, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_50_history = view.closes(lookback=self._window).select(
            pl.col("NIFTY 50").alias("nifty_50_close")
        )
        nifty_50_closes = [float(v) for v in nifty_50_history["nifty_50_close"].to_list()]

        symbol_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "NIFTY 50" not in nifty_50_history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            open_prices = [
                float(view.history(lookback=i + 1)["open"].item())
                for i, _ in enumerate(closes)
            ]
            daily_returns = [(c - o) / o for c, o in zip(closes, open_prices)]
            cumulative_return = sum(daily_returns)

            nifty_50_close_series = [
                float(nifty_50_closes[i]) if i < len(nifty_50_closes) else 0.0
                for i in range(len(closes))
            ]
            nifty_50_cumulative_return = sum(
                [(c - o) / o for c, o in zip(nifty_50_close_series, open_prices)]
            )

            if cumulative_return and nifty_50_cumulative_return:
                rs_value = cumulative_return / nifty_50_cumulative_return
                symbol_returns[symbol] = rs_value

        sorted_symbols = [
            s for _, s in sorted(symbol_returns.items(), key=lambda item: -item[1])
        ][: self._top_n]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest