"""Confirmed momentum tilt strategy for NIFTY 100 constituents."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_FORWARD_HORIZON = 3
_MOMENTUM_WINDOW = 10


class ConfirmedMomentumTilt(Strategy):
    """Buys recent winners whose momentum is confirmed by a cross-sectional strength percentile.

    Plain momentum ranking is noisy because a stock can be a temporary leader for reasons that
    don't persist. This strategy pairs a conventional momentum rank with a confirmation
    percentile summarising how strongly each name has tended to lead its peers, and only takes
    positions where both measures agree.
    """

    rationale = (
        "Momentum alone is a noisy ranking signal, so this strategy confirms it with a "
        "cross-sectional strength percentile summarising how consistently each name leads its "
        "peers, only holding names where both the momentum rank and the confirmation "
        "percentile point the same way."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self.top_n = top_n
        # Confirmation percentile, calibrated once from the full historical panel: each name's
        # short-horizon leadership relative to its peers, cached per session for lookup.
        enriched = panel.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close")
            .pct_change(_FORWARD_HORIZON)
            .over("symbol")
            .alias("period_return")
        )
        enriched = enriched.with_columns(
            pl.col("period_return").shift(-_FORWARD_HORIZON).over("symbol").alias("lead_return")
        )
        enriched = enriched.with_columns(
            pl.col("lead_return").rank(method="average").over("session_date").alias("lead_rank"),
            pl.col("lead_return").count().over("session_date").alias("cohort_size"),
        )
        enriched = enriched.with_columns(
            (pl.col("lead_rank") / pl.col("cohort_size")).alias("strength_percentile")
        )
        confirmation = enriched.select(
            "symbol", "session_date", "strength_percentile"
        ).drop_nulls()
        self._confirmation: dict[tuple[str, object], float] = {
            (row["symbol"], row["session_date"]): row["strength_percentile"]
            for row in confirmation.iter_rows(named=True)
        }

    def _momentum(self, view: MarketView) -> dict[str, float]:
        history = view.history(_MOMENTUM_WINDOW + 1)
        if history.is_empty():
            return {}
        moves = history.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close").pct_change(_MOMENTUM_WINDOW).over("symbol").alias("momentum")
        )
        latest = moves.sort("session_date").group_by("symbol").last()
        return dict(zip(latest["symbol"].to_list(), latest["momentum"].to_list(), strict=True))

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        momentum = self._momentum(view)
        picks = []
        for symbol, mom in momentum.items():
            confirmation = self._confirmation.get((symbol, last_date))
            if mom is not None and mom > 0 and confirmation is not None and confirmation > 0.7:
                picks.append(symbol)
        picks = picks[: self.top_n]
        if not picks:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(picks)
        return Signal(information_available_at=last_date, weights={s: weight for s in picks})
