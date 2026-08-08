from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain market movements are influenced by seasonal patterns. By identifying "
        "and exploiting these patterns, we can construct a strategy that performs better "
        "during specific times of the year."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_years * 252)  # Assuming 252 trading days per year
        if closes.height < self._lookback_years * 252:
            return Signal(information_available_at=stamp, weights={})

        latest_year = stamp.year
        seasonality_scores: dict[str, float] = {}

        for symbol in view.symbols:
            history = view.history(lookback=self._lookback_years)
            yearly_closes = history.select(
                pl.col("symbol") == symbol,
                pl.col("session_date").dt.year().alias("year"),
                pl.col("adj_close").alias("close"),
            ).group_by("year").agg(pl.col("close").mean().alias("avg_close"))

            recent_avg = yearly_closes.filter(pl.col("year") == latest_year).select(
                "avg_close"
            ).first().get("avg_close", None)

            if recent_avg is not None:
                scores = (
                    history.select(
                        pl.col("symbol") == symbol,
                        pl.col("session_date").dt.year().alias("year"),
                        (pl.col("adj_close") / pl.col("avg_close")) - 1.0
                    ).group_by("year")
                    .agg(pl.col("adj_close") / pl.col("avg_close"))
                    .sort("year", descending=True)
                    .select(
                        "year",
                        (pl.col("adj_close") / pl.col("avg_close")).alias("score"),
                    )
                )

                scores = (
                    scores.filter(pl.col("year").is_in(list(range(latest_year - self._lookback_years, latest_year + 1))))
                ).to_pandas()

                if not scores.empty:
                    recent_score = scores[scores["year"] == latest_year]["score"].iloc[0]
                    historical_max = max(scores["score"])
                    seasonality_scores[symbol] = (recent_score - historical_max) / historical_max

        ranked_symbols = sorted(seasonality_scores, key=seasonality_scores.get, reverse=True)[:5]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest