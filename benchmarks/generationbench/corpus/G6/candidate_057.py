from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy selects stocks with the lowest 60-day rolling standard deviation of "
        "daily log returns. By focusing on low-volatility assets, we aim to reduce overall portfolio risk while achieving more stable returns."
    )

    def __init__(self, window: int = 60, top_percentile: float = 0.25) -> None:
        self._window = window
        self._top_percentile = top_percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if f"adj_close@{s}" in history.columns]

        log_returns = (
            history.lazy()
            .filter(pl.col("symbol").is_in(symbols))
            .with_columns(
                (pl.col(f"adj_close@{s}") / pl.col(f"adj_close@{s}").shift(1) - 1.0).alias(f"log_return@{s}")
                for s in symbols
            )
            .group_by("session_date")
            .agg([
                (pl.col(f"log_return@{s}").std().alias(f"volatility@{s}") for s in symbols)
            ])
            .collect()
        )

        sorted_symbols = log_returns.sort([f"volatility@{s}" for s in symbols], descending=False)[0].to_dict(False)

        top_symbols = [
            symbol
            for i, (symbol, _) in enumerate(sorted_symbols.items())
            if i < int(len(symbols) * self._top_percentile)
        ]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest