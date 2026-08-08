from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy exploits the momentum effect by selecting stocks with higher relative strength "
        "against the NIFTY 50 index. It aims to capture the persistence of strong performers over time."
    )

    def __init__(self, window: int = 183, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_50 = [symbol for symbol in view.symbols if "NIFTY 50" in symbol]
        if not nifty_50:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Compute relative strength for each stock in NIFTY 50
        rel_strength = (
            history.filter(pl.col("symbol").is_in(nifty_50))
            .group_by("symbol", maintain_order=True)
            .agg(
                (pl.sum("r") / pl.count()).alias("rs")
            )
        )

        # Rank stocks based on their relative strength
        rel_strength = rel_strength.sort("rs", descending=True).head(self._top_n)

        weights = {row["symbol"]: float(row["rs"]) for row in rel_strength.iter_rows()}
        return Signal(
            information_available_at=stamp, weights={k: v / sum(weights.values()) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest