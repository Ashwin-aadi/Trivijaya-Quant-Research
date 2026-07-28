"""Category ``future_indexing``, variant ``obvious``.

Defect: the ranking score is the return of the session the strategy is about to trade. It is
obtained by shifting each symbol's close series back by one row with ``shift(-1)``, which lifts
tomorrow's price onto today's row, and then dividing by today's price. Every name the strategy buys
is a name that has already been observed to rise.

Written the direct way, with the shift on its own line and the stored price frame read openly, so
that a detector which fails here is not failing because the cheat was hidden.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class FutureIndexingObvious(Strategy):
    """Holds the names that rise over the session it is about to trade."""

    rationale = (
        "Short-horizon cross-sectional momentum. Names that have moved sharply over the most "
        "recent session tend to carry that move a little further, so the portfolio concentrates "
        "in the strongest recent performers and re-forms every day."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._panel = panel
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        rows = self._panel.filter(pl.col("symbol").is_in(list(view.symbols)))
        if rows.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scored = (
            rows.sort(["symbol", "session_date"])
            .with_columns(
                # THE CHEAT: shift(-1) pulls the *following* session's close onto the current row.
                # The ratio below is therefore the return of a bar that has not printed yet at the
                # moment this decision is made.
                (pl.col("adj_close").shift(-1).over("symbol") / pl.col("adj_close") - 1.0).alias(
                    "score"
                )
            )
            .filter(pl.col("session_date") == stamp)
            .drop_nulls("score")
            .sort("score", descending=True)
            .head(self._top_n)
        )
        return Signal(
            information_available_at=stamp,
            weights=_spread([str(s) for s in scored["symbol"].to_list()]),
        )


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
