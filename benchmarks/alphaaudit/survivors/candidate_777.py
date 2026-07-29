from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy that aims to capture higher returns by "
        "allocating more capital to assets with lower historical volatility. This approach "
        "is based on the empirical observation that low-volatility stocks tend to outperform."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback or history.width == 1:
            return Signal(information_available_at=stamp, weights={})

        volatility_by_symbol = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility")),
                pl.col("session_date").max().alias("last_session"),
            )
            .sort("volatility")
            .select(["symbol", "volatility", "last_session"])
        )

        if volatility_by_symbol.height < 1:
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = [row["symbol"] for row in volatility_by_symbol.to_dicts()]
        num_symbols = min(len(low_vol_symbols), 5)
        weights = {s: 1.0 / num_symbols for s in low_vol_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest