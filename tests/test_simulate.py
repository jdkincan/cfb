"""Offline tests for the season simulator.

The fixture is a small round-robin league with known strengths, so most
assertions have an arithmetic right answer rather than only checking that
nothing crashed.
"""

from __future__ import annotations

import pytest

from cfbmeta.simulate import (
    FCS_WIN_PROBABILITY,
    actual_win_totals,
    build_schedule,
    coverage_report,
    simulate,
)

STRENGTH = {"alpha": 20.0, "bravo": 7.0, "charlie": 0.0, "delta": -7.0, "echo": -20.0}
TEAMS = list(STRENGTH)
CONF = {t: "Test Conference" for t in TEAMS}


def schedule(with_results=False, fcs_opponent=False):
    games, gid = [], 500
    for i, home in enumerate(TEAMS):
        for away in TEAMS[i + 1:]:
            game = {
                "id": gid, "season": 2026, "week": 1,
                "homeTeam": home.title(), "awayTeam": away.title(),
                "homeConference": CONF[home], "awayConference": CONF[away],
                "conferenceGame": True, "neutralSite": False,
            }
            if with_results:
                better = STRENGTH[home] >= STRENGTH[away]
                game["homePoints"] = 30 if better else 10
                game["awayPoints"] = 10 if better else 30
            games.append(game)
            gid += 1
    if fcs_opponent:
        games.append({
            "id": 900, "season": 2026, "week": 2,
            "homeTeam": "Alpha", "awayTeam": "Tiny State",
            "homeConference": CONF["alpha"], "awayConference": "FCS",
            "conferenceGame": False, "neutralSite": False,
        })
    return games


def margin_of(game):
    """Projected home margin from the known strengths, no home edge."""
    from cfbmeta.ratings import normalize_team
    home = normalize_team(game["homeTeam"])
    away = normalize_team(game["awayTeam"])
    if home not in STRENGTH or away not in STRENGTH:
        return None
    return STRENGTH[home] - STRENGTH[away]


ELIGIBLE = set(TEAMS)


# -- schedule construction ---------------------------------------------------
def test_build_schedule_indexes_every_team():
    prepared, index, names, display = build_schedule(schedule(), margin_of, ELIGIBLE)
    assert len(prepared) == 10           # C(5,2)
    assert set(names) == set(TEAMS)
    assert display["alpha"] == "Alpha"   # original casing preserved for output


def test_build_schedule_locks_completed_games():
    prepared, *_ = build_schedule(schedule(with_results=True), margin_of, ELIGIBLE)
    assert all(g["outcome"] is not None for g in prepared)
    assert all(g["margin"] is None for g in prepared)


def test_build_schedule_routes_an_unrated_opponent_to_the_base_rate():
    prepared, _, names, _ = build_schedule(
        schedule(fcs_opponent=True), margin_of, ELIGIBLE
    )
    fcs = [g for g in prepared if g["fcs_side"] is not None]
    assert len(fcs) == 1
    assert fcs[0]["fcs_side"] == "home"   # Alpha is the rated side


def test_build_schedule_drops_a_game_with_no_eligible_team():
    games = [{
        "id": 1, "homeTeam": "Tiny State", "awayTeam": "Other Tiny",
        "homeConference": "FCS", "awayConference": "FCS",
    }]
    prepared, *_ = build_schedule(games, margin_of, ELIGIBLE)
    assert prepared == []


# -- simulation --------------------------------------------------------------
@pytest.fixture
def sim():
    return simulate(schedule(), margin_of, season=2026, sims=4000,
                    sigma_game=16.0, sigma_team=6.0, eligible=ELIGIBLE, seed=3)


def test_simulation_orders_teams_by_strength(sim):
    order = [t.team for t in sim.ranked()]
    assert order == TEAMS


def test_win_totals_average_to_the_games_played(sim):
    # Every game produces exactly one winner, so total wins = total games.
    total = sum(t.mean_wins for t in sim.teams.values())
    assert total == pytest.approx(10.0, abs=0.05)


def test_win_distribution_is_a_probability_distribution(sim):
    for outcome in sim.teams.values():
        assert sum(outcome.win_counts.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0 <= w <= outcome.scheduled for w in outcome.win_counts)


def test_probability_of_at_least_is_monotonic(sim):
    alpha = sim.teams["alpha"]
    values = [alpha.probability_of_at_least(w) for w in range(0, 5)]
    assert values == sorted(values, reverse=True)
    assert alpha.probability_of_at_least(0) == pytest.approx(1.0, abs=1e-6)


def test_interval_widens_with_the_level(sim):
    alpha = sim.teams["alpha"]
    narrow = alpha.interval(0.50)
    wide = alpha.interval(0.95)
    assert wide[0] <= narrow[0] and wide[1] >= narrow[1]


def test_completed_games_are_not_re_simulated():
    done = simulate(schedule(with_results=True), margin_of, season=2026, sims=500,
                    eligible=ELIGIBLE, seed=1)
    assert done.games_simulated == 0
    assert done.games_locked == 10
    # The stronger team won every game by construction, so wins are exact.
    assert done.teams["alpha"].mean_wins == pytest.approx(4.0)
    assert done.teams["echo"].mean_wins == pytest.approx(0.0)


def test_the_team_offset_widens_the_spread():
    """Without it, win totals are binomial and far too confident."""
    def spread(sigma_team):
        s = simulate(schedule(), margin_of, season=2026, sims=6000,
                     sigma_game=16.0, sigma_team=sigma_team,
                     eligible=ELIGIBLE, seed=5)
        low, high = s.teams["charlie"].interval(0.80)
        return high - low

    assert spread(6.0) >= spread(0.0)


def test_unrated_opponent_is_won_at_the_base_rate():
    with_fcs = simulate(schedule(fcs_opponent=True), margin_of, season=2026,
                        sims=8000, sigma_team=6.0, eligible=ELIGIBLE, seed=9)
    without = simulate(schedule(), margin_of, season=2026, sims=8000,
                       sigma_team=6.0, eligible=ELIGIBLE, seed=9)
    gain = with_fcs.teams["alpha"].mean_wins - without.teams["alpha"].mean_wins
    assert gain == pytest.approx(FCS_WIN_PROBABILITY, abs=0.03)
    assert with_fcs.games_base_rate == 1


def test_the_unrated_opponent_is_not_reported_as_a_team():
    sim = simulate(schedule(fcs_opponent=True), margin_of, season=2026, sims=500,
                   eligible=ELIGIBLE, seed=2)
    assert "tiny state" not in sim.teams


# -- conference titles -------------------------------------------------------
def test_conference_titles_sum_to_one(sim):
    total = sum(t.conference_title for t in sim.teams.values()
                if t.conference_title is not None)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_the_best_team_is_the_title_favourite(sim):
    best = max(sim.teams.values(), key=lambda t: t.conference_title or 0)
    assert best.team == "alpha"


def test_a_conference_with_no_conference_games_has_no_title():
    """Independents all finish 0-0; 'most conference wins' is a coin flip."""
    games = schedule()
    for game in games:
        game["conferenceGame"] = False
        game["homeConference"] = game["awayConference"] = "FBS Independents"
    sim = simulate(games, margin_of, season=2026, sims=500,
                   eligible=ELIGIBLE, seed=4)
    assert all(t.conference_title is None for t in sim.teams.values())


# -- calibration helpers -----------------------------------------------------
def test_actual_win_totals_counts_wins_over_unrated_opponents():
    games = schedule(with_results=True, fcs_opponent=True)
    games[-1]["homePoints"] = 49
    games[-1]["awayPoints"] = 3
    totals = actual_win_totals(games, eligible=ELIGIBLE)
    assert totals["alpha"] == 5          # 4 league wins + the FCS game
    assert "tiny state" not in totals


def test_coverage_report_measures_what_it_claims(sim):
    truth = {t: round(o.mean_wins) for t, o in sim.teams.items()}
    report = coverage_report(sim, truth, 0.80)
    assert report["n"] == len(sim.teams)
    assert report["coverage"] == pytest.approx(1.0)   # truth set to the centre
    assert report["mae"] < 0.6


def test_coverage_report_on_no_overlap(sim):
    report = coverage_report(sim, {"nobody": 5}, 0.80)
    assert report["n"] == 0


def test_simulating_an_empty_schedule_is_safe():
    result = simulate([], margin_of, season=2026, sims=100)
    assert result.teams == {}
    assert result.sims == 0
