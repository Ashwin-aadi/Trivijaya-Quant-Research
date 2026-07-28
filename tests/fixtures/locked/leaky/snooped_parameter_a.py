"""Cross-sectional z-score reversion on NIFTY constituents.

Buys names whose price has fallen further below their trailing average than a chosen number of
standard deviations, on the premise that such dislocations are usually followed by a partial
recovery. The entry threshold is calibrated once, up front, by checking which trigger level has
historically produced the strongest subsequent bounce.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_CANDIDATE_THRESHOLDS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)


class ZScoreReversion(Strategy):
    """Buys names whose z-score below their trailing mean crosses a calibrated threshold."""

    rationale = (
        "Names that dislocate sharply below their recent trading range tend to see at least a "
        "partial mean reversion over the following sessions, provided the move is not so "
        "extreme that it reflects a genuine change in fundamentals. Calibrating the trigger "
        "level against history should give a threshold that actually separates noise from a "
        "tradeable dislocation, rather than guessing a round number."
    )

    def __init__(self, calibration_data: pl.DataFrame, window: int = 20, top_n: int = 8) -> None:
        self._window = window
        self._top_n = top_n
        self._threshold = self._calibrate_threshold(calibration_data, window)

    @staticmethod
    def _calibrate_threshold(panel: pl.DataFrame, window: int) -> float:
        sorted_panel = panel.sort(["symbol", "session_date"])
        rolling_mean = pl.col("adj_close").rolling_mean(window).over("symbol")
        rolling_std = pl.col("adj_close").rolling_std(window).over("symbol")
        with_stats = sorted_panel.with_columns(
            [rolling_mean.alias("mean"), rolling_std.alias("std")]
        )
        z_score = ((pl.col("adj_close") - pl.col("mean")) / pl.col("std")).alias("z")
        forward_close = pl.col("adj_close").shift(-5).over("symbol").alias("fwd_close")
        scored = with_stats.with_columns([z_score, forward_close])
        best_threshold, best_score = _CANDIDATE_THRESHOLDS[0], float("-inf")
        for threshold in _CANDIDATE_THRESHOLDS:
            triggers = scored.filter(pl.col("z") <= -threshold)
            if triggers.is_empty():
                continue
            forward_return = (triggers["fwd_close"] / triggers["adj_close"] - 1.0).mean()
            score = float(forward_return) if forward_return is not None else float("-inf")
            if score > best_score:
                best_threshold, best_score = threshold, score
        return best_threshold

    def generate(self, view: MarketView) -> Signal:
        history = view.history(self._window)
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        stats = history.group_by("symbol").agg(
            [pl.col("adj_close").mean().alias("mean"), pl.col("adj_close").std().alias("std")]
        )
        latest = view.latest_close()
        names = stats["symbol"].to_list()
        means = stats["mean"].to_list()
        stds = stats["std"].to_list()
        candidates: list[tuple[str, float]] = []
        for symbol, mean, std in zip(names, means, stds, strict=True):
            price = latest.get(symbol)
            if price is None or not std:
                continue
            z = (price - mean) / std
            if z <= -self._threshold:
                candidates.append((symbol, z))
        candidates.sort(key=lambda row: row[1])
        chosen = candidates[: self._top_n]
        weights = {symbol: 1.0 / len(chosen) for symbol, _ in chosen} if chosen else {}
        return Signal(information_available_at=view.as_of, weights=weights)
