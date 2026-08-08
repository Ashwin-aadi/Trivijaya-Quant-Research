from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the persistence of excess returns in highly liquid stocks by "
        "equal-weighting them. High liquidity often implies lower transaction costs and more "
        "efficient price discovery, leading to potentially higher average returns."
    )

    def __init__(self, window: int = 200) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by the symbols that are in the latest closes
        symbols = [symbol for symbol in view.symbols if symbol in view.closes().columns]

        # Calculate daily turnover or volume as a liquidity metric
        liquidity_metric = history.select(
            pl.col("symbol"), (pl.col("adj_close") * pl.col("volume")).alias("turnover")
        ).group_by("symbol").agg(pl.sum("turnover").alias("total_turnover"))

        # Rank symbols by their total turnover over the lookback period
        ranked_liquidity = liquidity_metric.sort("total_turnover", descending=True)

        # Select top 50% most liquid stocks based on the turnover metric
        n_symbols = len(symbols)
        if n_symbols <= 1:
            return Signal(information_available_at=stamp, weights={})

        n_top_symbols = int(n_symbols * 0.5)
        selected_symbols = ranked_liquidity.head(n_top_symbols)["symbol"].to_list()

        # Equal weight these symbols
        weight = 2.0 / (n_top_symbols * 100)  # Ensure no single stock contributes more than 2% to the risk

        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest