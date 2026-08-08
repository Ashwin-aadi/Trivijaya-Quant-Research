from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Dispersion20d(Strategy):
    rationale = (
        "This strategy exploits periods of high dispersion in sectoral returns by focusing on "
        "outliers within sectors. High dispersion often indicates heterogeneous market conditions, "
        "potentially offering higher returns through focused investment."
    )

    def __init__(self, window: int = 60, num_sectors: int = 5) -> None:
        self._window = window
        self._num_sectors = num_sectors

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sectoral_returns = self._compute_sectoral_returns(history)
        sectors_with_high_dispersion = self._identify_high_dispersions(sectoral_returns)

        if not sectors_with_high_dispersion:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sectors_with_high_dispersion)
        return Signal(
            information_available_at=stamp, weights={
                sector: weight for sector in sectors_with_high_dispersion
            }
        )

    def _compute_sectoral_returns(self, history: pl.DataFrame) -> pl.DataFrame:
        symbols = view.symbols
        sectoral_grouped = (
            history.groupby("symbol").agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
        ).collect()
        return sectoral_grouped

    def _identify_high_dispersions(self, returns: pl.DataFrame) -> list[str]:
        sectors = []
        for symbol in view.symbols:
            if symbol not in returns.columns:
                continue
            values = [float(v) for v in returns[symbol].to_list()]
            mean_return = sum(values) / len(values)
            std_dev = (sum((v - mean_return) ** 2 for v in values) / len(values)) ** 0.5
            cv = std_dev / mean_return if mean_return != 0 else 0
            if cv > 1:
                sectors.append(symbol)

        return sorted(sectors, key=lambda x: returns[x].std().item(), reverse=True)[: self._num_sectors]

def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest