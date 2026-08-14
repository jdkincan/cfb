import math

import pytest

from cfbmeta.probability import (
    MarginDistribution,
    american_to_decimal,
    american_to_payout,
    evaluate_spread_bet,
    evaluate_total_bet,
    expected_value,
    implied_probability,
    kelly_fraction,
    normal_cdf,
    remove_vig,
)


def test_normal_cdf_known_values():
    assert normal_cdf(0) == pytest.approx(0.5)
    assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_margin_distribution_is_normalized_and_skips_zero():
    dist = MarginDistribution(7.0, 16.0)
    assert sum(dist.pmf.values()) == pytest.approx(1.0)
    assert 0 not in dist.pmf


def test_margin_distribution_mean_tracks_mu():
    # Key-number reweighting is symmetric enough that the mean stays close.
    dist = MarginDistribution(10.0, 16.0)
    assert dist.mean() == pytest.approx(10.0, abs=0.6)


def test_key_numbers_lift_three_and_seven():
    dist = MarginDistribution(0.0, 16.0)
    # 3 and 7 should carry more mass than their neighbours despite the
    # normal density being nearly flat across that range.
    assert dist.pmf[3] > dist.pmf[2]
    assert dist.pmf[3] > dist.pmf[4]
    assert dist.pmf[7] > dist.pmf[6]
    assert dist.pmf[7] > dist.pmf[8]


def test_home_and_away_cover_are_complementary_with_push():
    dist = MarginDistribution(3.5, 16.0)
    line = 7.0
    total = dist.p_home_cover(line) + dist.p_away_cover(line) + dist.p_push(line)
    assert total == pytest.approx(1.0)


def test_half_point_line_has_no_push():
    dist = MarginDistribution(3.0, 16.0)
    assert dist.p_push(6.5) == 0.0
    assert dist.p_push(7.0) > 0.0


def test_win_probability_moves_the_right_way():
    favored = MarginDistribution(14.0, 16.0).p_home_win()
    even = MarginDistribution(0.0, 16.0).p_home_win()
    dog = MarginDistribution(-14.0, 16.0).p_home_win()
    assert favored > even > dog
    assert even == pytest.approx(0.5, abs=0.05)


def test_pricing_helpers():
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_payout(-110) == pytest.approx(0.9090909)
    assert implied_probability(-110) == pytest.approx(0.5238, abs=1e-3)


def test_remove_vig_sums_to_one():
    a, b = remove_vig(-110, -110)
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.5)


def test_expected_value_break_even_at_vig():
    # 52.38% is break-even at -110.
    assert expected_value(0.5238, -110) == pytest.approx(0.0, abs=1e-3)
    assert expected_value(0.55, -110) > 0
    assert expected_value(0.50, -110) < 0


def test_kelly_zero_without_edge():
    assert kelly_fraction(0.50, -110) == 0.0
    assert kelly_fraction(0.5238, -110) == pytest.approx(0.0, abs=1e-3)
    assert kelly_fraction(0.60, -110) > 0


def test_kelly_ignores_push_mass():
    # A push is no-action, so it should not be counted as a loss.
    with_push = kelly_fraction(0.55, -110, p_push=0.08)
    without_push = kelly_fraction(0.55 / 0.92, -110)
    assert with_push == pytest.approx(without_push, abs=1e-6)


class TestSpreadEvaluation:
    def test_home_side_when_model_likes_home(self):
        # Model says home by 10, market says home by 3 -> take home.
        bet = evaluate_spread_bet(10.0, -3.0, 16.0, min_edge_points=1.5)
        assert bet.side == "home"
        assert bet.edge_points == pytest.approx(7.0)
        assert bet.cover_probability > 0.5
        assert bet.units > 0

    def test_away_side_when_model_likes_away(self):
        # Model says home by 1, market says home by 10 -> take away.
        bet = evaluate_spread_bet(1.0, -10.0, 16.0, min_edge_points=1.5)
        assert bet.side == "away"
        assert bet.edge_points == pytest.approx(-9.0)
        assert bet.cover_probability > 0.5

    def test_no_play_when_edge_below_threshold(self):
        bet = evaluate_spread_bet(3.5, -3.0, 16.0, min_edge_points=1.5)
        assert bet.side == "none"
        assert bet.units == 0.0
        assert bet.is_play is False

    def test_no_play_without_a_line(self):
        bet = evaluate_spread_bet(7.0, None, 16.0)
        assert bet.side == "none"

    def test_underdog_spread_sign_is_reported_from_bet_side(self):
        # Home is a 10-point dog (+10) and we like home.
        bet = evaluate_spread_bet(-3.0, 10.0, 16.0, min_edge_points=1.5)
        assert bet.side == "home"
        assert bet.line == pytest.approx(10.0)

    def test_units_are_capped(self):
        bet = evaluate_spread_bet(40.0, -3.0, 16.0, max_units=3.0, bankroll_units=100.0)
        assert bet.units <= 3.0

    def test_edge_is_symmetric_around_the_line(self):
        # Home favored by 3, model says home by 10 -> +7 toward home.
        a = evaluate_spread_bet(10.0, -3.0, 16.0)
        # Mirror image: home is a 3-point dog, model says home loses by 10.
        b = evaluate_spread_bet(-10.0, 3.0, 16.0)
        assert a.edge_points == pytest.approx(7.0)
        assert b.edge_points == pytest.approx(-7.0)
        assert a.side == "home"
        assert b.side == "away"

    def test_small_edge_against_the_number_is_not_a_play(self):
        # Market has home +3, model has home losing by 4: a 1-point edge.
        bet = evaluate_spread_bet(-4.0, 3.0, 16.0, min_edge_points=1.5)
        assert bet.edge_points == pytest.approx(-1.0)
        assert bet.side == "none"


class TestTotalEvaluation:
    def test_over_when_projection_is_higher(self):
        bet = evaluate_total_bet(60.0, 52.0, 13.5, min_edge_points=2.5)
        assert bet.side == "over"
        assert bet.cover_probability > 0.5

    def test_under_when_projection_is_lower(self):
        bet = evaluate_total_bet(45.0, 55.0, 13.5, min_edge_points=2.5)
        assert bet.side == "under"

    def test_no_play_on_small_total_edge(self):
        bet = evaluate_total_bet(53.0, 52.0, 13.5, min_edge_points=2.5)
        assert bet.side == "none"

    def test_missing_inputs_are_safe(self):
        assert evaluate_total_bet(None, 52.0, 13.5).side == "none"
        assert evaluate_total_bet(52.0, None, 13.5).side == "none"
