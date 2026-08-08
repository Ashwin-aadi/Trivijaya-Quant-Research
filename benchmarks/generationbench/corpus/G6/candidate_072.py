from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy leverages the tendency of stocks with recent strong performance to continue outperforming in the short term. "
        "By selecting top-performing stocks relative to the NIFTY 50 index and equally weighting them, we aim for a balanced portfolio that maximizes momentum benefits."
    )

    def __init__(self, window: int = 20, lookback: int = 60, top_n: int = 7) -> None:
        self._window = window
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50 = history.select(
            pl.col("symbol").filter(pl.col("symbol").is_in(view.symbols))
        )
        closes = view.closes(lookback=self._window + self._lookback)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate momentum for each stock
        momentum: pl.DataFrame = (
            nifty50.select(
                [
                    pl.col("symbol"),
                    (pl.col("adj_close").shift(-self._lookback) / pl.col("adj_close").shift(self._window - 1) - 1.0).alias("momentum")
                ]
            ).with_columns((pl.col("momentum") * 100).alias("momentum_pct"))
        )

        # Rank stocks by momentum
        ranked = (
            momentum.with_column(pl.col("symbol").cast(pl.Utf8))
            .group_by("symbol")
            .agg(pl.col("momentum_pct").mean().alias("avg_momentum"))
            .sort("avg_momentum", descending=True)
            .head(self._top_n)
        )

        symbols = [row[0] for row in ranked.to_dict(as_pandas=False)["symbol"]]
        weights = {s: 0.04 for s in symbols}

        return Signal(
            information_available_at=stamp, weights={k: float(v) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest