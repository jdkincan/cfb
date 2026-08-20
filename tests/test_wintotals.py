"""Tests for season win-total pricing."""

from __future__ import annotations

import pytest

from cfbmeta.wintotals import (
    WinTotalBet,
    WinTotalLine,
    build_card,
    evaluate,
    load_lines,
    outcome_probabilities,
    smeared_cdf,
)


class FakeOutcome:
    def __init__(self, team, counts, display="", conference=""):
        self.team = team
        self.display = display or team.title()
        self.conference = conference
        self.win_counts = counts
        self.mean_wins = sum(w * p for w, p in counts.items())


def symmetric(centre=7, spread=3):
    """A tidy symmetric distribution centred on ``centre``."""
    raw = {w: 1.0 / (1 + abs(w - centre)) for w in range(centre - spread, centre + spread + 1)}
    total = sum(raw.values())
    return {w: p / total for w, p in raw.items()}


class FakeSim:
    def __init__(self, teams):
        self.teams = {t.team: t for t in teams}


# -- the distribution --------------------------------------------------------
def test_cdf_runs_from_zero_to_one():
    counts = symmetric()
    assert smeared_cdf(counts, -5) == pytest.approx(0.0)
    assert smeared_cdf(counts, 20) == pytest.approx(1.0)


def test_cdf_is_monotone():
    counts = symmetric()
    values = [smeared_cdf(counts, x / 4) for x in range(0, 60)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_half_lines_cannot_push():
    over, under, push = outcome_probabilities(symmetric(), 7.5)
    assert push == 0.0
    assert over + under == pytest.approx(1.0)


def test_whole_lines_carry_real_push_mass():
    """Roughly an eighth of a season's outcomes land exactly on the number."""
    over, under, push = outcome_probabilities(symmetric(centre=7), 7.0)
    assert push > 0.05
    assert over + under + push == pytest.approx(1.0)


def test_a_symmetric_team_is_a_coin_flip_on_its_own_whole_number():
    """At 7.0 the tails are equal; the middle is push."""
    over, under, push = outcome_probabilities(symmetric(centre=7), 7.0)
    assert over == pytest.approx(under, abs=0.02)
    assert push > 0.0


def test_a_half_line_above_the_centre_favours_the_under():
    """7.5 puts the whole 7-win mass on the under, so it is not a coin flip."""
    over, under, _ = outcome_probabilities(symmetric(centre=7), 7.5)
    assert under > over


def test_shifting_up_raises_the_over():
    base, _, _ = outcome_probabilities(symmetric(), 7.5)
    lifted, _, _ = outcome_probabilities(symmetric(), 7.5, shift=1.0)
    assert lifted > base


# -- pricing -----------------------------------------------------------------
def test_a_large_disagreement_becomes_a_play():
    outcome = FakeOutcome("alpha", symmetric(centre=10))
    bet = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0)
    assert bet.is_play
    assert bet.side == "over"
    assert bet.units > 0


def test_agreement_with_the_number_is_a_pass():
    outcome = FakeOutcome("alpha", symmetric(centre=7))
    bet = evaluate(outcome, WinTotalLine("Alpha", 7.0))
    assert not bet.is_play
    assert any("threshold" in n for n in bet.notes)


def test_shrink_scales_the_edge_toward_the_posted_number():
    outcome = FakeOutcome("alpha", symmetric(centre=10))
    full = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0)
    part = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=0.35)
    assert full.edge_wins == pytest.approx(2.5)
    assert part.edge_wins == pytest.approx(2.5 * 0.35)
    assert part.p_over < full.p_over


def test_zero_shrink_prices_the_market_number_and_never_bets():
    outcome = FakeOutcome("alpha", symmetric(centre=10))
    bet = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=0.0)
    assert bet.edge_wins == pytest.approx(0.0)
    assert not bet.is_play


def test_the_under_is_taken_when_the_model_is_low():
    outcome = FakeOutcome("alpha", symmetric(centre=4))
    bet = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0)
    assert bet.side == "under"


def test_a_worse_price_can_kill_an_otherwise_live_bet():
    outcome = FakeOutcome("alpha", symmetric(centre=9))
    fair = evaluate(outcome, WinTotalLine("Alpha", 7.5, over_price=-110), shrink=0.5)
    gouged = evaluate(outcome, WinTotalLine("Alpha", 7.5, over_price=-400), shrink=0.5)
    assert fair.is_play
    assert gouged.expected_value < fair.expected_value


def test_the_threshold_is_enforced():
    outcome = FakeOutcome("alpha", symmetric(centre=8))
    loose = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0, min_edge_wins=0.1)
    strict = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0, min_edge_wins=2.0)
    assert loose.is_play
    assert not strict.is_play


def test_units_respect_the_cap():
    outcome = FakeOutcome("alpha", symmetric(centre=12))
    bet = evaluate(outcome, WinTotalLine("Alpha", 5.5), shrink=1.0, max_units=1.5)
    assert bet.units <= 1.5


def test_a_team_with_no_distribution_is_skipped():
    bet = evaluate(FakeOutcome("alpha", {}), WinTotalLine("Alpha", 7.5))
    assert not bet.is_play
    assert any("No simulated distribution" in n for n in bet.notes)


def test_win_probability_follows_the_chosen_side():
    outcome = FakeOutcome("alpha", symmetric(centre=10))
    bet = evaluate(outcome, WinTotalLine("Alpha", 7.5), shrink=1.0)
    assert bet.win_probability == bet.p_over


# -- the card ----------------------------------------------------------------
def test_card_prices_every_posted_total():
    sim = FakeSim([FakeOutcome("alpha", symmetric(centre=10)),
                   FakeOutcome("bravo", symmetric(centre=4))])
    card = build_card(sim, [WinTotalLine("Alpha", 7.5), WinTotalLine("Bravo", 7.5)],
                      shrink=1.0)
    assert len(card) == 2
    assert {b.side for b in card} == {"over", "under"}


def test_card_ignores_a_team_that_was_not_simulated():
    sim = FakeSim([FakeOutcome("alpha", symmetric())])
    assert build_card(sim, [WinTotalLine("Nowhere State", 7.5)]) == []


def test_card_scales_down_to_the_exposure_cap():
    sim = FakeSim([FakeOutcome(f"t{i}", symmetric(centre=11)) for i in range(6)])
    lines = [WinTotalLine(f"T{i}", 6.5) for i in range(6)]
    card = build_card(sim, lines, shrink=1.0, max_total_units=4.0)
    staked = sum(b.units for b in card if b.is_play)
    assert staked == pytest.approx(4.0, abs=0.06)


def test_card_leaves_a_small_slate_alone():
    sim = FakeSim([FakeOutcome("alpha", symmetric(centre=10))])
    card = build_card(sim, [WinTotalLine("Alpha", 7.5)], shrink=1.0, max_total_units=50.0)
    assert card[0].units > 0


def test_card_puts_plays_before_passes():
    sim = FakeSim([FakeOutcome("alpha", symmetric(centre=7)),
                   FakeOutcome("bravo", symmetric(centre=11))])
    card = build_card(sim, [WinTotalLine("Alpha", 7.0), WinTotalLine("Bravo", 7.5)],
                      shrink=1.0)
    assert card[0].is_play and not card[-1].is_play


# -- loading -----------------------------------------------------------------
def test_load_accepts_a_bare_number(tmp_path):
    p = tmp_path / "w.yml"
    p.write_text("totals:\n  Arkansas: 4.5\n")
    lines = load_lines(p)
    assert lines[0].team == "Arkansas" and lines[0].total == 4.5
    assert lines[0].over_price == -110


def test_load_accepts_prices_and_a_book(tmp_path):
    p = tmp_path / "w.yml"
    p.write_text("book: FanDuel\ntotals:\n  Georgia:\n    total: 10.5\n"
                 "    over: -140\n    under: 115\n")
    line = load_lines(p)[0]
    assert (line.total, line.over_price, line.under_price) == (10.5, -140, 115)
    assert line.book == "FanDuel"


def test_load_skips_a_blank_entry(tmp_path):
    p = tmp_path / "w.yml"
    p.write_text("totals:\n  Arkansas: 4.5\n  Georgia:\n")
    assert len(load_lines(p)) == 1


def test_load_rejects_a_malformed_entry(tmp_path):
    p = tmp_path / "w.yml"
    p.write_text("totals:\n  Arkansas: [1, 2]\n")
    with pytest.raises(ValueError):
        load_lines(p)


def test_missing_file_is_not_an_error(tmp_path):
    assert load_lines(tmp_path / "absent.yml") == []


def test_line_key_is_normalized():
    assert WinTotalLine("Ohio State", 8.5).key == "ohio state"


def test_bet_defaults_to_no_play():
    assert not WinTotalBet(team="x").is_play
