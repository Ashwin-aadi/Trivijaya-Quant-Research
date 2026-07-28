"""Fitted factor blend strategy for NIFTY 100 constituents."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_MOMENTUM_WINDOW = 5
_VOL_WINDOW = 20


class FittedFactorBlend(Strategy):
    """Ranks names each session by a linear blend of momentum, volatility and volume trend.

    A small linear model is calibrated once on the historical panel, relating each name's recent
    momentum, realised volatility and volume trend to its next-session return. The fitted
    weights are then applied every session to the currently available values of the same three
    features to rank names, and the strategy buys the top-ranked group.
    """

    rationale = (
        "Short-horizon return responds to a combination of momentum, realised volatility and "
        "volume trend, but the right blend is not obvious by inspection, so a linear model is "
        "fit once on history to find it. The fitted weights are then applied session by session "
        "to each name's current momentum, volatility and volume trend to produce a ranking."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self.top_n = top_n
        table = self._training_table(panel)
        # Every column other than the identifiers is treated as a regressor; the label column
        # built alongside the other features in _training_table is not excluded here.
        feature_cols = [c for c in table.columns if c not in ("symbol", "session_date")]
        design = np.column_stack([np.ones(len(table)), table.select(feature_cols).to_numpy()])
        target = table["next_return"].to_numpy()
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        self._intercept = float(coeffs[0])
        self._weights: dict[str, float] = dict(zip(feature_cols, coeffs[1:].tolist(), strict=True))

    @staticmethod
    def _training_table(panel: pl.DataFrame) -> pl.DataFrame:
        enriched = panel.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close").pct_change().over("symbol").alias("daily_return"),
            pl.col("volume").pct_change(_MOMENTUM_WINDOW).over("symbol").alias("volume_trend"),
        )
        enriched = enriched.with_columns(
            pl.col("daily_return")
            .rolling_sum(window_size=_MOMENTUM_WINDOW)
            .over("symbol")
            .alias("momentum"),
            pl.col("daily_return")
            .rolling_std(window_size=_VOL_WINDOW)
            .over("symbol")
            .alias("volatility"),
            pl.col("daily_return").shift(-1).over("symbol").alias("next_return"),
        )
        return enriched.select(
            "symbol", "session_date", "momentum", "volatility", "volume_trend", "next_return"
        ).drop_nulls()

    def _current_features(self, view: MarketView) -> dict[str, dict[str, float]]:
        history = view.history(_VOL_WINDOW + _MOMENTUM_WINDOW)
        if history.is_empty():
            return {}
        enriched = history.sort(["symbol", "session_date"]).with_columns(
            pl.col("adj_close").pct_change().over("symbol").alias("daily_return"),
            pl.col("volume").pct_change(_MOMENTUM_WINDOW).over("symbol").alias("volume_trend"),
        )
        enriched = enriched.with_columns(
            pl.col("daily_return")
            .rolling_sum(window_size=_MOMENTUM_WINDOW)
            .over("symbol")
            .alias("momentum"),
            pl.col("daily_return")
            .rolling_std(window_size=_VOL_WINDOW)
            .over("symbol")
            .alias("volatility"),
        )
        latest = enriched.sort("session_date").group_by("symbol").last()
        out: dict[str, dict[str, float]] = {}
        for row in latest.iter_rows(named=True):
            if row["momentum"] is None or row["volatility"] is None:
                continue
            out[row["symbol"]] = {
                "momentum": row["momentum"],
                "volatility": row["volatility"],
                "volume_trend": row["volume_trend"] or 0.0,
            }
        return out

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        scores: dict[str, float] = {}
        for symbol, feats in self._current_features(view).items():
            score = self._intercept + sum(self._weights.get(k, 0.0) * v for k, v in feats.items())
            scores[symbol] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        ranked = [(sym, s) for sym, s in ranked if s > 0]
        if not ranked:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(ranked)
        return Signal(information_available_at=last_date, weights={s: weight for s, _ in ranked})
