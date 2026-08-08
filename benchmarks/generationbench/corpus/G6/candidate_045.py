from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "The strategy focuses on reducing risk by tilting towards stocks with lower historical volatility. "
        "Historical volatility is measured using the 20-day rolling standard deviation of daily returns, and a "
        "bottom 30% selection process ensures diversification while favoring more stable assets."
    )

    def __init__(self, window: int = 20, rank_threshold: float = 0.7) -> None:
        self._window = window
        self._rank_threshold = rank_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty() or history.height < 20 * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Compute rolling standard deviation of daily returns
        history = history.with_columns(
            (
                pl.col("r").rolling_std(window_size=self._window, min_periods=20)
                .alias(f"std_{self._window}")
            )
        )

        # Rank symbols by their 20-day rolling std
        ranked = history.groupby("symbol", maintain_order=True).agg(
            (pl.col(f"std_{self._window}").mean().alias("rolling_std_mean"))
        ).sort("rolling_std_mean")

        if ranked.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter top 70% and bottom 30% symbols
        cutoff = int(len(ranked) * (1 - self._rank_threshold))
        low_vol_symbols = [r["symbol"] for r in ranked[cutoff:].to_dicts()]

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate inverse volatility scores and normalize
        vol_scores = (
            history.filter(pl.col("symbol").is_in(low_vol_symbols))
            .group_by("symbol")
            .agg(
                (pl.col(f"std_{self._window}").mean().alias("rolling_std_mean"))
            )
            .select(
                pl.col("symbol"),
                1.0 / pl.col("rolling_std_mean").rank(method="ordinal", descending=False).alias("inv_vol_score"),
            )
        )

        # Rebalance monthly
        if (view.as_of - stamp).days > 30:
            return Signal(information_available_at=stamp, weights={})

        inv_scores = vol_scores.sort("inv_vol_score", descending=True)["inv_vol_score"].to_list()
        total_weight = sum(inv_scores)
        weight = {s: w / total_weight for s, w in zip(vol_scores["symbol"], inv_scores)}

        return Signal(
            information_available_at=stamp,
            weights={s: weight[s] for s in view.symbols if s in weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest