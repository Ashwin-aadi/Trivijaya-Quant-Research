"""Extreme single-session move fade strategy for NIFTY 100 constituents."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class ExtremeMoveFade(Strategy):
    """Fades unusually large single-day moves.

    Retail-driven order flow in Indian mid-caps tends to overreact to news, pushing prices past
    a fair reaction on the day of the move and giving some of it back over the following
    sessions. A session's return is flagged as "extreme" when it falls outside a name's usual
    range of daily moves, and the strategy takes the opposite side, expecting partial reversion.
    """

    rationale = (
        "Single-session moves that fall outside a stock's usual range of daily returns are "
        "disproportionately driven by short-term flow rather than a durable change in "
        "fundamentals, and Indian mid-caps have tended to give back part of an outsized move "
        "over the following few sessions. This strategy buys names after an unusually sharp "
        "decline and avoids names after an unusually sharp rally."
    )

    def __init__(self, panel: pl.DataFrame, tail: float = 0.05, max_positions: int = 8) -> None:
        self.max_positions = max_positions
        # Each name's usual tail bands, calibrated once from the historical return panel so
        # the same reference bands can be reused for every session's fade decision.
        returns = panel.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close").pct_change().over("symbol").alias("daily_return")
        )
        bands = returns.group_by("symbol").agg(
            pl.col("daily_return").quantile(tail).alias("lower_band"),
            pl.col("daily_return").quantile(1 - tail).alias("upper_band"),
        )
        self._lower_band: dict[str, float] = dict(
            zip(bands["symbol"].to_list(), bands["lower_band"].to_list(), strict=True)
        )
        self._upper_band: dict[str, float] = dict(
            zip(bands["symbol"].to_list(), bands["upper_band"].to_list(), strict=True)
        )

    def _last_session_return(self, view: MarketView) -> dict[str, float]:
        recent = view.history(2)
        if recent.is_empty():
            return {}
        recent = recent.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close").pct_change().over("symbol").alias("daily_return")
        )
        latest = recent.group_by("symbol").last()
        return dict(zip(latest["symbol"].to_list(), latest["daily_return"].to_list(), strict=True))

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        moves = self._last_session_return(view)
        candidates = []
        for symbol, move in moves.items():
            lower = self._lower_band.get(symbol)
            if lower is not None and move is not None and move <= lower:
                candidates.append(symbol)
        candidates = candidates[: self.max_positions]
        if not candidates:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(candidates)
        return Signal(information_available_at=last_date, weights={s: weight for s in candidates})
