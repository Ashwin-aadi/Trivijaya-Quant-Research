"""Category ``target_in_features``, variant ``buried``.

Defect: the predicted quantity enters the feature set through a monotone transform rather than
directly. ``_signed_strength`` returns the signed square root of the forward return, and that value
is added to the feature dictionary under the name ``conviction``. A square root of the absolute
value, sign preserved, is strictly increasing, so it carries the full ordering of the label. The
model is ranking on the answer with a different shape.

The transform is the thing worth having a fixture for. A check that compares feature columns
against the label column by name or by equality finds nothing here: ``conviction`` is not the label
and does not equal the label. It is a bijection of it. Everything else in the file is an ordinary
feature-engineering pipeline with a winsoriser, a coverage filter and a ranking blend, which is
where a transform like this hides in real code.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class ConvictionWeightedRanker(Strategy):
    """Ranks names on a blend of engineered features and a conviction term."""

    rationale = (
        "Ranking on a single characteristic throws away the information in the others, and "
        "averaging characteristics on their raw scales lets whichever one has the widest "
        "dispersion dominate. Winsorising each characteristic, ranking it within the "
        "cross-section, and averaging the ranks makes the blend scale-free, and a conviction "
        "term lets the book concentrate when the signals agree strongly."
    )

    def __init__(
        self,
        engineered: pl.DataFrame,
        winsor: float = 0.02,
        top_n: int = 12,
    ) -> None:
        self._engineered = engineered
        self._winsor = winsor
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        block = self._engineered.filter(
            (pl.col("session_date") == stamp) & (pl.col("symbol").is_in(list(view.symbols)))
        )
        if block.is_empty():
            return Signal(information_available_at=stamp, weights={})

        features = self._collect(block)
        if not features:
            return Signal(information_available_at=stamp, weights={})

        blended = _blend_ranks(features)
        chosen = sorted(blended, key=lambda s: (-blended[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))

    def _collect(self, block: pl.DataFrame) -> dict[str, dict[str, float]]:
        """Assemble the feature dictionary for every name with complete coverage."""
        out: dict[str, dict[str, float]] = {}
        for row in block.iter_rows(named=True):
            momentum = row.get("momentum_126")
            reversal = row.get("reversal_5")
            quality = row.get("accruals")
            if momentum is None or reversal is None or quality is None:
                continue
            out[str(row["symbol"])] = {
                "momentum": self._winsorise(float(momentum)),
                "reversal": self._winsorise(float(reversal)),
                "quality": self._winsorise(float(quality)),
                # THE CHEAT: `conviction` is a signed square root of the forward return, which is
                # the quantity this model exists to predict. The transform is strictly increasing,
                # so it preserves the ordering of the label exactly; adding it to the feature set
                # means the ranking below is the label's own ranking wearing a different scale.
                "conviction": _signed_strength(float(row.get("realised_fwd_5") or 0.0)),
            }
        return out

    def _winsorise(self, value: float) -> float:
        """Clip a standardised characteristic into a fixed band to limit single-name influence."""
        limit = 1.0 - self._winsor
        return max(min(value, limit), -limit)


def _signed_strength(value: float) -> float:
    """Square root of magnitude with the sign preserved; compresses tails, keeps the ordering."""
    return math.copysign(math.sqrt(abs(value)), value)


def _blend_ranks(features: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average cross-sectional rank across every feature, scaled to the unit interval."""
    names = sorted(features)
    if not names:
        return {}
    columns = sorted(features[names[0]])
    total: dict[str, float] = dict.fromkeys(names, 0.0)
    for column in columns:
        ordered = sorted(names, key=lambda s: (features[s].get(column, 0.0), s))
        for rank, symbol in enumerate(ordered):
            total[symbol] += rank / max(len(names) - 1, 1)
    return {symbol: value / len(columns) for symbol, value in total.items()}


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
