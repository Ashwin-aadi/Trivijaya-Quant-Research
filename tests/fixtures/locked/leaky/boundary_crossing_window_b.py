"""Monthly strength tilt strategy for NIFTY 100 constituents."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class MonthlyStrengthTilt(Strategy):
    """Tilts toward names showing the strongest average return for the calendar month.

    Short-term daily noise obscures which names are genuinely strong for the month, so this
    strategy summarises each name's performance at the monthly level and tilts weight toward the
    stocks with the best monthly showing, refreshed once the calendar month changes.
    """

    rationale = (
        "Monthly institutional rebalancing flows in India tend to reinforce whichever names are "
        "already showing the strongest performance for that month, so a name's average return "
        "for the current calendar month is a useful measure of which way that flow is pushing "
        "it. This strategy tilts toward the names with the best monthly average return."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 12) -> None:
        self.top_n = top_n
        # Each name's monthly return is summarised once from the full historical panel and
        # then looked up by calendar month for every session that falls within it.
        with_month = panel.sort(["symbol", "session_date"]).with_columns(
            pl.col("session_date").dt.year().alias("year"),
            pl.col("session_date").dt.month().alias("month"),
            pl.col("adj_close").pct_change().over("symbol").alias("daily_return"),
        )
        monthly = with_month.group_by(["symbol", "year", "month"]).agg(
            pl.col("daily_return").mean().alias("monthly_strength")
        )
        self._monthly_strength: dict[tuple[str, int, int], float] = {
            (row["symbol"], row["year"], row["month"]): row["monthly_strength"]
            for row in monthly.iter_rows(named=True)
            if row["monthly_strength"] is not None
        }

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            strength = self._monthly_strength.get((symbol, last_date.year, last_date.month))
            if strength is not None:
                scores[symbol] = strength
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        ranked = [(sym, s) for sym, s in ranked if s > 0]
        if not ranked:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(ranked)
        return Signal(information_available_at=last_date, weights={s: weight for s, _ in ranked})
