"""Cross-sectional valuation-band rotation strategy for NIFTY 100 constituents."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class ValuationBandRotation(Strategy):
    """Rotates into names trading unusually cheap relative to their typical price level.

    Large- and mid-cap Indian equities tend to oscillate around a level that holds for years at
    a time rather than drifting indefinitely; when the current price sits well below a stock's
    customary range it has historically pulled back toward it. The strategy standardises each
    name's current price against its own typical trading level and buys the names sitting
    furthest below it.
    """

    rationale = (
        "Large-cap Indian equities revert to a persistent valuation range over multi-year "
        "horizons rather than trending without bound. Names trading well below their customary "
        "price level are statistically cheap and have tended to close the gap, so this strategy "
        "buys the decile of names furthest below their typical level and holds cash otherwise."
    )

    def __init__(self, panel: pl.DataFrame, entry_z: float = 1.0, top_decile: int = 10) -> None:
        self.entry_z = entry_z
        self.top_decile = top_decile
        # Typical price level and spread, calibrated once from the historical panel supplied
        # at construction so the same reference points can be reused every session.
        stats = panel.group_by("symbol").agg(
            pl.col("adj_close").mean().alias("typical_level"),
            pl.col("adj_close").std().alias("typical_spread"),
        )
        self._typical_level: dict[str, float] = dict(
            zip(stats["symbol"].to_list(), stats["typical_level"].to_list(), strict=True)
        )
        self._typical_spread: dict[str, float] = dict(
            zip(stats["symbol"].to_list(), stats["typical_spread"].to_list(), strict=True)
        )

    def _z_scores(self, view: MarketView) -> dict[str, float]:
        scores: dict[str, float] = {}
        for symbol, price in view.latest_close().items():
            spread = self._typical_spread.get(symbol)
            level = self._typical_level.get(symbol)
            if not spread or level is None:
                continue
            scores[symbol] = (price - level) / spread
        return scores

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        last_date = history["session_date"].max()
        scores = self._z_scores(view)
        cheap = sorted(scores.items(), key=lambda kv: kv[1])[: self.top_decile]
        cheap = [(sym, z) for sym, z in cheap if z <= -self.entry_z]
        if not cheap:
            return Signal(information_available_at=last_date)
        weight = 1.0 / len(cheap)
        weights = {sym: weight for sym, _ in cheap}
        return Signal(information_available_at=last_date, weights=weights)
