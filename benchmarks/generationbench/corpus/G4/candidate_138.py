from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks outperforming the broader market index based on relative "
        "strength. Strong performers tend to continue outperforming due to factors like positive "
        "earnings surprises or favorable news. The portfolio is rebalanced quarterly."
    )

    def __init__(self, window: int = 60, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        nifty_50_close = history.filter(pl.col("symbol") == "NIFTY50").select(
            pl.col("adj_close")
        )
        if nifty_50_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_data: dict[str, pl.DataFrame] = {}
        for symbol in view.symbols:
            if "NIFTY50" not in nifty_50_close.columns:
                continue
            history_symbol = history.filter(pl.col("symbol") == symbol)
            if history_symbol.height < self._window or history_symbol.is_empty():
                continue

            adj_close = history_symbol.select(pl.col("adj_close")).to_series().to_list()
            nifty_50_adj_close = (
                nifty_50_close.to_series().to_list() * len(adj_close)
            )  # Repeat NIFTY50 close for each day
            returns = [(a - b) / b if b != 0 else 0.0 for a, b in zip(adj_close, adj_close[:-1])]
            nifty_returns = [
                (a - b) / b if b != 0 else 0.0 for a, b in zip(nifty_50_adj_close, nifty_50_adj_close[:-1])
            ]
            cumulative_return = sum(
                [(r + 1) * (n + 1) - 1 for r, n in zip(returns, nifty_returns)]
            )
            symbol_data[symbol] = cumulative_return

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        ranked_symbols = sorted(symbol_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in ranked_symbols[: self._top_n]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest