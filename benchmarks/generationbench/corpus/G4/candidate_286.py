from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalJanuaryEffect(Strategy):
    rationale = (
        "Historically, the Indian equity market shows January effects where certain stocks or indices exhibit above-average returns. "
        "This strategy capitalizes on these historical patterns by increasing exposure to selected securities during January."
    )

    def __init__(self, lookback_years: int = 10, max_positions: int = 30, max_position_weight: float = 0.05) -> None:
        self._lookback_years = lookback_years
        self._max_positions = max_positions
        self._max_position_weight = max_position_weight

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)
        if history.height < 252 * self._lookback_years:
            return Signal(information_available_at=stamp, weights={})

        # Filter data to only include January returns
        january_returns = history.filter(pl.col("session_date").dt.month() == 1).select(
            pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(252) - 1.0).alias("jan_return")
        )

        # Rank symbols based on their average January returns
        ranked_symbols = january_returns.groupby("symbol").agg(
            (pl.col("jan_return").mean().rank(method="ordinal", descending=True)).alias("rank")
        ).sort("rank")

        if ranked_symbols.height < self._max_positions:
            top_symbols = [row["symbol"] for row in ranked_symbols.iter_rows()]
        else:
            top_symbols = [row["symbol"] for row in ranked_symbols.head(self._max_positions).iter_rows()]

        # Assign weights based on rank
        weights = {s: max(0.01, self._max_position_weight * (self._max_positions - i + 1) / self._max_positions)
                   for i, s in enumerate(top_symbols)}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest