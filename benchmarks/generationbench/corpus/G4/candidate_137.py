from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies breakout continuation patterns in Indian equity markets. "
        "It leverages technical indicators like Bollinger Bands to confirm breakouts and then "
        "enters positions during subsequent consolidations within the new range, capitalizing on "
        "momentum and investor behavior."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.select(["symbol", "session_date", "close"]).filter(
                (pl.col("symbol") == symbol)
            ).sort("session_date")

            # Identify breakout points using Bollinger Bands
            mean = df.select(pl.col("close").mean()).item()
            std_dev = df.select(pl.col("close").std()).item()
            upper_band = mean + 2 * std_dev
            lower_band = mean - 2 * std_dev

            breakout_points = []
            for i in range(len(df) - 1):
                if (df["close"][i] < lower_band and df["close"][i + 1] > lower_band) or \
                   (df["close"][i] > upper_band and df["close"][i + 1] < upper_band):
                    breakout_points.append(i)

            # Look for continuation patterns after breakouts
            if not breakout_points:
                continue

            last_breakout_idx = max(breakout_points)
            post_breakout_data = df[last_breakout_idx:].collect()

            if post_breakout_data.height < 20:  # Ensure enough data for consolidation check
                continue

            closing_prices = [float(v) for v in post_breakout_data["close"].to_list()]
            consolidation_high, consolidation_low = max(closing_prices), min(closing_prices)

            # Find potential continuation entries within the new range
            candidates = []
            for i in range(last_breakout_idx + 1, len(df)):
                if df["close"][i] < consolidation_high and df["close"][i] > consolidation_low:
                    candidates.append(i)
                    break

            candidate_scores = {symbol: abs(df["close"][idx] - consolidation_high) for idx in candidates}

            # Rank candidates based on score
            ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])
            picks.extend([c for c, _ in ranked_candidates[: self._top_n]])

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest