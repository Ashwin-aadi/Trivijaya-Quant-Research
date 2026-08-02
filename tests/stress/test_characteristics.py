"""Characteristics must describe books that are constructed to have a known answer.

Every test here builds a book whose holding period, concentration or persistence can be worked out
by hand, so a failure points at the arithmetic rather than at a judgement call.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stress.characteristics import (
    book_autocorrelation,
    concentration,
    holding_period,
    turnover_profile,
    univariate_betas,
)


def test_an_equally_weighted_book_of_four_names_has_a_herfindahl_of_one_quarter() -> None:
    """Four equal shares of 0.25 each: HHI = 4 x 0.25^2 = 0.25, effective breadth exactly 4."""
    books = [{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}] * 10
    result = concentration(books)
    assert result["mean_herfindahl"] == pytest.approx(0.25)
    assert result["effective_holdings"] == pytest.approx(4.0)
    assert result["mean_n_holdings"] == pytest.approx(4.0)
    assert result["cash_session_rate"] == 0.0


def test_a_single_name_book_is_maximally_concentrated() -> None:
    result = concentration([{"A": 1.0}] * 5)
    assert result["mean_herfindahl"] == pytest.approx(1.0)
    assert result["mean_largest_weight_share"] == pytest.approx(1.0)


def test_a_short_position_counts_toward_concentration_by_its_absolute_size() -> None:
    """A book long 0.5 and short 0.5 is two bets, not zero. Netting them would report no book."""
    result = concentration([{"A": 0.5, "B": -0.5}] * 4)
    assert result["mean_herfindahl"] == pytest.approx(0.5)
    assert result["mean_n_holdings"] == pytest.approx(2.0)


def test_cash_sessions_are_counted_rather_than_scored() -> None:
    """A session holding nothing must not be given a concentration value it does not have."""
    books = [{"A": 1.0}, {}, {"A": 1.0}, {}]
    result = concentration(books)
    assert result["cash_session_rate"] == pytest.approx(0.5)
    assert result["mean_herfindahl"] == pytest.approx(1.0)   # from the two held sessions only


def test_a_name_held_throughout_gives_a_holding_period_of_the_whole_sample() -> None:
    """One entry, ten held sessions: 10 / 1 = 10."""
    result = holding_period([{"A": 1.0}] * 10)
    assert result["n_entries"] == 1.0
    assert result["mean_holding_period"] == pytest.approx(10.0)


def test_a_book_churned_every_session_gives_a_holding_period_of_one() -> None:
    """Alternating between two names: 10 entries over 10 held sessions."""
    books = [{"A": 1.0} if i % 2 == 0 else {"B": 1.0} for i in range(10)]
    result = holding_period(books)
    assert result["n_entries"] == 10.0
    assert result["mean_holding_period"] == pytest.approx(1.0)


def test_a_re_entered_name_counts_as_two_entries_not_one() -> None:
    """Held, exited, held again is two holdings of a name, not one long one."""
    books = [{"A": 1.0}, {"A": 1.0}, {}, {"A": 1.0}]
    result = holding_period(books)
    assert result["n_entries"] == 2.0
    assert result["mean_holding_period"] == pytest.approx(1.5)   # 3 held sessions / 2 entries


def test_a_residual_weight_below_the_threshold_is_not_a_holding() -> None:
    """A 1e-18 leftover in an exited name would otherwise register as a continuous position."""
    books = [{"A": 1.0}, {"A": 1e-18}, {"A": 1.0}]
    result = holding_period(books)
    assert result["n_entries"] == 2.0


def test_an_unchanging_book_has_similarity_one_at_every_horizon() -> None:
    books = [{"A": 0.5, "B": 0.5}] * 100
    result = book_autocorrelation(books, horizons=(5, 21))
    assert result["book_similarity_5d"] == pytest.approx(1.0)
    assert result["book_similarity_21d"] == pytest.approx(1.0)
    assert result["book_similarity_5d_n"] == 95.0


def test_a_book_that_alternates_between_disjoint_names_has_zero_similarity_at_odd_lags() -> None:
    """Disjoint books share no name, so their dot product — and the similarity — is exactly zero."""
    books = [{"A": 1.0} if i % 2 == 0 else {"B": 1.0} for i in range(60)]
    result = book_autocorrelation(books, horizons=(5, 21))
    assert result["book_similarity_5d"] == pytest.approx(0.0)   # 5 is odd: always disjoint
    assert result["book_similarity_21d"] == pytest.approx(0.0)


def test_similarity_reports_how_many_comparisons_it_actually_made() -> None:
    """No number without its sample size: a horizon longer than the sample must say so."""
    result = book_autocorrelation([{"A": 1.0}] * 10, horizons=(21,))
    assert result["book_similarity_21d_n"] == 0.0
    assert np.isnan(result["book_similarity_21d"])


def test_turnover_profile_reports_the_share_of_sessions_that_traded() -> None:
    result = turnover_profile([1.0, 0.0, 0.0, 1.0])
    assert result["mean_turnover"] == pytest.approx(0.5)
    assert result["trading_session_rate"] == pytest.approx(0.5)


def test_a_beta_against_a_factor_that_is_the_series_itself_is_one() -> None:
    """The identity case. Anything other than 1.0 means the covariance is mis-scaled."""
    rng = np.random.default_rng(42)
    series = np.asarray(rng.normal(0.0, 0.01, 500))
    result = univariate_betas(series, {"itself": series})
    assert result["beta_itself"] == pytest.approx(1.0)


def test_a_beta_against_a_doubled_factor_is_one_half() -> None:
    rng = np.random.default_rng(7)
    series = np.asarray(rng.normal(0.0, 0.01, 500))
    result = univariate_betas(series, {"doubled": 2.0 * series})
    assert result["beta_doubled"] == pytest.approx(0.5)


def test_a_constant_factor_yields_no_beta_rather_than_a_division_by_zero() -> None:
    """A factor with no variance cannot explain anything, and must not return an infinity."""
    rng = np.random.default_rng(1)
    result = univariate_betas(
        np.asarray(rng.normal(0.0, 0.01, 100)), {"flat": np.zeros(100)}
    )
    assert np.isnan(result["beta_flat"])
