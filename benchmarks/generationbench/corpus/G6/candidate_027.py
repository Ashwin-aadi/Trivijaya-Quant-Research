from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy balances equal weighting with liquidity screening to optimize portfolio performance while mitigating risks associated with illiquid stocks. By ensuring that highly liquid, well-performing companies receive appropriate weightings, we aim to capture market-wide opportunities effectively."
    )

    def __init__(self, min_turnover_threshold: float = 100_000, window: int = 20) -> None:
        self._min_turnover_threshold = min_turnover_threshold
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily turnover for each symbol
        turnover_df = (
            history.groupby("symbol")
                   .agg(pl.col("volume").sum().alias("turnover"))
                   .with_columns(
                       (pl.col("turnover") / pl.col("turnover").mean()).alias("turnover_ratio")
                   )
        )

        # Filter symbols based on turnover ratio
        filtered_symbols = turnover_df.filter(
            (pl.col("turnover_ratio") >= self._min_turnover_threshold) &
            (pl.col("turnover_ratio") <= 1.5 * pl.col("turnover_ratio").mean())
        ).select(["symbol"])

        if filtered_symbols.height < 15:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in filtered_symbols["symbol"].to_list():
                continue
            picks.append(symbol)

        # Ensure we have at least 15-20 liquid stocks
        picks = picks[:min(20, len(picks))]

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest