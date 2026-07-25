"""Hold every name in the investable universe at equal weight."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class EqualWeightUniverse(Strategy):
    """The simplest possible portfolio: own everything, equally."""

    rationale = (
        "Owning the whole liquid universe at equal weight captures broad market exposure without "
        "expressing any view. It serves as the neutral baseline every other strategy is judged "
        "against."
    )

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=latest_visible(view),
            weights=equal_weight(view.symbols),
        )
