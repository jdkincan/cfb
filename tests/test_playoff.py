"""Tests for the twelve-team playoff simulation.

The synthetic league below has four conferences of eight, a full conference
round robin and a few cross-over games, with strengths assigned so the
favourites are known in advance. That gives the structural rules something to
be checked against rather than only being checked for not crashing.
"""

from __future__ import annotations

import pytest

from cfbmeta.playoff import (
    AUTO_BIDS,
    BYES,
    FIELD_SIZE,
    ROUNDS,
    PlayoffOdds,
    _bracket_pairs,
    committee_score,
    simulate_playoff,
    strength_of_schedule,
)

CONFS = ["Alpha", "Beta", "Gamma", "Delta"]


def _league(n_conf=4):
    ratings, games, gid = {}, [], 0
    confs = CONFS[:n_conf]
    for c, conf in enumerate(confs):
        for i in range(8):
            ratings[f"{conf}{i}"] = 20.0 - 2.5 * i - 1.5 * c
    for conf in confs:
        members = [f"{conf}{i}" for i in range(8)]
        for i, home in enumerate(members):
            for away in members[i + 1:]:
                gid += 1
                games.append({"id": gid, "week": 1 + gid % 12,
                              "homeTeam": home, "awayTeam": away,
                              "homeConference": conf, "awayConference": conf,
                              "conferenceGame": True})
    for i in range(8):
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
            if a >= n_conf or b >= n_conf:
                continue
            gid += 1
            games.append({"id": gid, "week": 13,
                          "homeTeam": f"{confs[a]}{i}", "awayTeam": f"{confs[b]}{i}",
                          "homeConference": confs[a], "awayConference": confs[b],
                          "conferenceGame": False})
    return ratings, games


def _run(sims=3000, seed=3, **kw):
    ratings, games = _league()
    return simulate_playoff(
        games, lambda g: ratings[g["homeTeam"]] - ratings[g["awayTeam"]] + 2.7,
        ratings, season=2026, sims=sims, eligible=set(ratings), seed=seed, **kw)


@pytest.fixture(scope="module")
def sim():
    return _run()


# -- the committee model -----------------------------------------------------
def test_a_loss_costs_more_than_a_win_pays():
    """The single most important thing the fit learned about this committee."""
    base = committee_score(10, 2, 20.0, 5.0)
    extra_win = committee_score(11, 2, 20.0, 5.0) - base
    extra_loss = base - committee_score(10, 3, 20.0, 5.0)
    assert extra_win > 0
    assert extra_loss > extra_win * 2


def test_score_rises_with_rating_and_schedule():
    assert committee_score(10, 2, 25.0, 5.0) > committee_score(10, 2, 20.0, 5.0)
    assert committee_score(10, 2, 20.0, 8.0) > committee_score(10, 2, 20.0, 5.0)


def test_undefeated_beats_one_loss_at_equal_quality():
    assert committee_score(13, 0, 18.0, 4.0) > committee_score(12, 1, 18.0, 4.0)


# -- the bracket -------------------------------------------------------------
def test_first_round_pairs_five_twelve_through_eight_nine():
    field = list(range(12))  # index i == seed i+1
    assert _bracket_pairs(field) == [(4, 11), (5, 10), (6, 9), (7, 8)]


def test_top_four_seeds_are_absent_from_the_first_round():
    field = list(range(12))
    playing = {t for pair in _bracket_pairs(field) for t in pair}
    assert playing.isdisjoint(set(field[:BYES]))


# -- conservation ------------------------------------------------------------
def test_exactly_twelve_teams_make_the_field(sim):
    assert sum(t.make_field for t in sim.teams.values()) == pytest.approx(FIELD_SIZE)


def test_exactly_four_byes(sim):
    assert sum(t.bye for t in sim.teams.values()) == pytest.approx(BYES)


@pytest.mark.parametrize("round_name,expected",
                         [("field", 12), ("quarterfinal", 8),
                          ("semifinal", 4), ("final", 2), ("champion", 1)])
def test_each_round_holds_the_right_number_of_teams(sim, round_name, expected):
    total = sum(t.reach[round_name] for t in sim.teams.values())
    assert total == pytest.approx(expected)


def test_every_seed_is_awarded_exactly_once_per_simulation(sim):
    for seed in range(1, FIELD_SIZE + 1):
        total = sum(t.seed_counts.get(seed, 0.0) for t in sim.teams.values())
        assert total == pytest.approx(1.0), f"seed {seed}"


def test_one_champion_per_conference(sim):
    assert sum(t.conference_title for t in sim.teams.values()) == pytest.approx(len(CONFS))


def test_bids_split_into_automatic_and_at_large(sim):
    autos = sum(t.auto_bid for t in sim.teams.values())
    larges = sum(t.at_large for t in sim.teams.values())
    # Only four conferences exist here, so only four automatic bids can be
    # issued even though the format allows five.
    assert autos == pytest.approx(len(CONFS))
    assert autos + larges == pytest.approx(FIELD_SIZE)


def test_automatic_bids_are_capped_at_five():
    ratings, games = _league()
    sim = simulate_playoff(
        games, lambda g: ratings[g["homeTeam"]] - ratings[g["awayTeam"]] + 2.7,
        ratings, season=2026, sims=500, eligible=set(ratings), seed=5)
    assert sum(t.auto_bid for t in sim.teams.values()) <= AUTO_BIDS + 1e-9


# -- behaviour ---------------------------------------------------------------
def test_the_best_team_is_the_most_likely_champion(sim):
    best = max(sim.teams.values(), key=lambda t: t.reach["champion"])
    assert best.team == "alpha0"


def test_byes_track_title_odds(sim):
    """Sanity on the bracket: seeds 1-4 skip a game, and it should show."""
    rows = [t for t in sim.teams.values() if t.make_field > 0.01]
    byes = [t.bye for t in rows]
    titles = [t.reach["champion"] for t in rows]
    mb, mt = sum(byes) / len(byes), sum(titles) / len(titles)
    cov = sum((b - mb) * (x - mt) for b, x in zip(byes, titles))
    sb = sum((b - mb) ** 2 for b in byes) ** 0.5
    st = sum((x - mt) ** 2 for x in titles) ** 0.5
    assert cov / (sb * st) > 0.8
    # And the team that earns byes most often wins it most often.
    assert (max(sim.teams.values(), key=lambda t: t.bye).team
            == max(sim.teams.values(), key=lambda t: t.reach["champion"]).team)


def test_probabilities_never_exceed_their_parent_round(sim):
    for t in sim.teams.values():
        for tighter, looser in zip(ROUNDS[1:], ROUNDS):
            assert t.reach[tighter] <= t.reach[looser] + 1e-9
        assert t.bye <= t.make_field + 1e-9


def test_mean_seed_is_conditional_on_making_the_field(sim):
    for t in sim.teams.values():
        if t.mean_seed is not None:
            assert 1.0 <= t.mean_seed <= FIELD_SIZE


def test_mean_seed_is_none_without_any_appearances():
    assert PlayoffOdds(team="x").mean_seed is None


def test_even_the_worst_team_can_back_into_the_field(sim):
    """It only has to win its conference, so the floor is above zero."""
    worst = min(sim.teams.values(), key=lambda t: t.make_field)
    assert 0.0 < worst.make_field < 0.05


def test_every_bid_is_either_automatic_or_at_large(sim):
    for t in sim.teams.values():
        assert t.auto_bid + t.at_large == pytest.approx(t.make_field)


# -- input handling ----------------------------------------------------------
def test_display_names_are_normalized_before_lookup():
    """Regression: unnormalized keys silently rated every team 0.0."""
    ratings, games = _league()
    loud = {k.upper(): v for k, v in ratings.items()}
    sim = simulate_playoff(
        games, lambda g: ratings[g["homeTeam"]] - ratings[g["awayTeam"]] + 2.7,
        loud, season=2026, sims=200, eligible={k.upper() for k in ratings}, seed=1)
    assert sum(t.make_field for t in sim.teams.values()) == pytest.approx(FIELD_SIZE)


def test_no_rated_team_returns_an_empty_simulation():
    _, games = _league()
    sim = simulate_playoff(games, lambda g: 0.0, {}, season=2026, sims=100,
                           eligible={"nobody"})
    assert sim.sims == 0 and not sim.teams


def test_an_empty_schedule_is_not_fatal():
    sim = simulate_playoff([], lambda g: 0.0, {"a": 1.0}, season=2026, sims=10)
    assert sim.sims == 0


def test_independents_win_no_conference_title_and_no_automatic_bid():
    ratings, games = _league()
    ratings["Lonely"] = 30.0  # strongest team in the league, but no conference
    for i in range(6):
        games.append({"id": 90000 + i, "week": 2 + i, "homeTeam": "Lonely",
                      "awayTeam": f"Alpha{i}", "homeConference": "",
                      "awayConference": "Alpha", "conferenceGame": False})
    sim = simulate_playoff(
        games, lambda g: ratings[g["homeTeam"]] - ratings[g["awayTeam"]] + 2.7,
        ratings, season=2026, sims=1500, eligible=set(ratings), seed=7)
    lonely = sim.teams["lonely"]
    assert lonely.conference_title == 0.0
    assert lonely.auto_bid == 0.0
    # It should still be a heavy at-large, being the best team on the board.
    assert lonely.make_field > 0.5
    assert lonely.at_large == pytest.approx(lonely.make_field)


# -- helpers -----------------------------------------------------------------
def test_strength_of_schedule_averages_opponent_ratings():
    schedule = [{"home": 0, "away": 1}, {"home": 0, "away": 2}]
    sos = strength_of_schedule(schedule, ["a", "b", "c"], [0.0, 10.0, 20.0])
    assert sos[0] == pytest.approx(15.0)
    assert sos[1] == pytest.approx(0.0)


def test_strength_of_schedule_survives_a_team_with_no_games():
    sos = strength_of_schedule([], ["a"], [5.0])
    assert sos[0] == pytest.approx(0.0)


def test_odds_expose_a_title_shortcut():
    odds = PlayoffOdds(team="x", reach={"champion": 0.25})
    assert odds.title() == 0.25
