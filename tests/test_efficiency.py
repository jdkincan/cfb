"""Offline tests for the weekly opponent-adjusted efficiency rating.

The fixture builds a small synthetic league with known team strengths, so the
fit has a right answer to be checked against rather than only being checked
for not crashing.
"""

from __future__ import annotations

import pytest

from cfbmeta.sources.efficiency import (
    EfficiencyRatings,
    fit_efficiency,
    game_efficiency,
    home_map,
    team_trend,
    weekly_series,
)

# Synthetic league: efficiency is exactly off[team] - def[opponent], so a
# correct fit must recover the ordering.
# Nine teams, because a double round robin has to clear the 60-observation
# floor the real fit imposes (9 teams -> 72 games -> 144 team-games).
STRENGTH = {
    "alpha": 0.45,
    "bravo": 0.32,
    "charlie": 0.21,
    "delta": 0.10,
    "echo": 0.00,
    "foxtrot": -0.12,
    "golf": -0.24,
    "hotel": -0.35,
    "india": -0.48,
}
TEAMS = list(STRENGTH)


def _schedule():
    """Double round robin across 4 weeks, alternating hosts."""
    games, stats, gid = [], [], 1000
    week = 1
    for first in (True, False):
        for i, home in enumerate(TEAMS):
            for away in TEAMS[i + 1:]:
                if not first:
                    home, away = away, home
                margin = (STRENGTH[home] - STRENGTH[away]) * 40 + 3
                games.append({
                    "id": gid, "season": 2025, "week": week,
                    "homeTeam": home.title(), "awayTeam": away.title(),
                    "homePoints": 24 + round(margin), "awayPoints": 24,
                    "neutralSite": False,
                })
                for team, opp in ((home, away), (away, home)):
                    stats.append({
                        "gameId": gid, "season": 2025, "week": week,
                        "team": team.title(), "opponent": opp.title(),
                        "offense": {
                            "ppa": STRENGTH[team] - STRENGTH[opp],
                            "successRate": 0.43,
                            "explosiveness": 1.2,
                        },
                        "defense": {"ppa": 0.0, "successRate": 0.43},
                    })
                gid += 1
                week = week % 4 + 1
    return games, stats


@pytest.fixture
def league():
    return _schedule()


# -- the composite -----------------------------------------------------------
def test_game_efficiency_combines_the_three_metrics():
    value = game_efficiency({"ppa": 0.5, "successRate": 0.5, "explosiveness": 2.0})
    assert value == pytest.approx(0.60 * 0.5 + 0.90 * 0.5 + 0.10 * 2.0)


def test_game_efficiency_needs_ppa_and_success_rate():
    assert game_efficiency({"successRate": 0.5}) is None
    assert game_efficiency({"ppa": 0.2}) is None
    assert game_efficiency(None) is None


def test_game_efficiency_tolerates_a_missing_explosiveness():
    assert game_efficiency({"ppa": 0.1, "successRate": 0.4}) is not None


def test_game_efficiency_reads_snake_case():
    assert game_efficiency({"ppa": 0.1, "success_rate": 0.4}) is not None


# -- the fit -----------------------------------------------------------------
def test_fit_recovers_the_known_ordering(league):
    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025)
    order = [team for team, _ in ratings.ranked()]
    assert order == TEAMS


def test_fit_produces_a_points_scale(league):
    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025)
    assert ratings.points_per_unit > 0
    assert ratings.points
    # Points and quality must agree on ordering.
    assert max(ratings.points, key=ratings.points.get) == "alpha"


def test_fit_recovers_home_field(league):
    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025)
    # The synthetic margin carries a +3 home edge.
    assert ratings.home_field == pytest.approx(3.0, abs=1.0)


def test_ratings_are_centred_on_zero(league):
    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025)
    assert sum(ratings.quality.values()) == pytest.approx(0.0, abs=1e-9)


def test_fit_returns_none_below_the_observation_floor(league):
    games, stats = league
    assert fit_efficiency(stats[:10], games, season=2025) is None


def test_before_week_is_strict(league):
    """Week 3 must be fit on weeks 1-2 only, or the backtest leaks."""
    games, stats = league
    ratings = fit_efficiency(stats, games, before_week=3, season=2025)
    assert ratings is None or ratings.observations <= sum(
        1 for r in stats if r["week"] < 3
    )


def test_fit_ignores_another_season(league):
    games, stats = league
    for row in stats:
        row["season"] = 2024
    assert fit_efficiency(stats, games, season=2025) is None


def test_fit_skips_games_missing_from_the_schedule(league):
    """Without home/away the home edge would land in team quality."""
    games, stats = league
    ratings = fit_efficiency(stats, games[:1], season=2025)
    assert ratings is None


# -- centring ----------------------------------------------------------------
def test_centring_on_a_subset_shifts_the_zero(league):
    games, stats = league
    strong = {"alpha", "bravo", "charlie"}
    ratings = fit_efficiency(stats, games, season=2025, center_on=strong)
    average = sum(ratings.quality[t] for t in strong) / len(strong)
    assert average == pytest.approx(0.0, abs=1e-9)
    # Everyone outside the reference population now sits below zero.
    assert ratings.quality["echo"] < 0


def test_a_reference_pool_that_has_not_played_yields_no_rating(league):
    """Refusing beats falling back: a mis-centred rating corrupts the blend.

    Early in a season only a handful of the reference teams have played. A
    rating centred on whoever happens to be in the fit is not on the same
    scale as SP+, and publishing it would silently shift every projection.
    """
    games, stats = league
    assert fit_efficiency(stats, games, season=2025, center_on={"nobody"}) is None


def test_partial_coverage_below_the_floor_also_refuses(league):
    games, stats = league
    # Two of nine reference teams present is under the coverage floor.
    pool = set(TEAMS) | {f"ghost{i}" for i in range(30)}
    assert fit_efficiency(stats, games, season=2025, center_on=pool) is None


def test_full_coverage_still_centres_normally(league):
    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025, center_on=set(TEAMS))
    assert sum(ratings.quality.values()) == pytest.approx(0.0, abs=1e-9)


def test_ranks_are_quoted_against_the_eligible_pool(league):
    games, stats = league
    pool = set(TEAMS) - {"alpha"}
    ratings = fit_efficiency(stats, games, season=2025, center_on=pool)
    # Alpha is the best team but is outside the pool, so it is not ranked.
    assert [t for t, _ in ratings.ranked()] == TEAMS[1:]
    assert ratings.rank_of("alpha") is None
    assert ratings.rank_of("bravo") == 1
    assert [t for t, _ in ratings.ranked(include_all=True)][0] == "alpha"


def test_rank_of_an_unknown_team_is_none(league):
    games, stats = league
    assert fit_efficiency(stats, games, season=2025).rank_of("nowhere") is None


# -- the series --------------------------------------------------------------
def test_weekly_series_is_keyed_by_week(league):
    games, stats = league
    series = weekly_series(stats, games, [3, 4, 5], season=2025)
    assert set(series) <= {3, 4, 5}
    assert all(isinstance(v, EfficiencyRatings) for v in series.values())
    for week, ratings in series.items():
        assert ratings.week == week


def test_weekly_series_grows_its_sample(league):
    games, stats = league
    series = weekly_series(stats, games, [4, 5], season=2025)
    if 4 in series and 5 in series:
        assert series[5].observations > series[4].observations


def test_team_trend_reports_week_rating_and_rank(league):
    games, stats = league
    series = weekly_series(stats, games, [4, 5], season=2025)
    trend = team_trend(series, "Alpha")
    assert trend
    for week, value, rank in trend:
        assert week in series
        assert rank >= 1
        assert isinstance(value, float)


def test_team_trend_is_empty_for_an_unknown_team(league):
    games, stats = league
    series = weekly_series(stats, games, [5], season=2025)
    assert team_trend(series, "Nowhere State") == []


# -- plumbing ----------------------------------------------------------------
def test_home_map_records_neutral_sites():
    mapping = home_map([
        {"id": 1, "homeTeam": "Alpha", "awayTeam": "Bravo", "neutralSite": True},
        {"id": 2, "homeTeam": "Charlie", "awayTeam": "Delta"},
    ])
    assert mapping[1] == ("alpha", "bravo", True)
    assert mapping[2] == ("charlie", "delta", False)


def test_home_map_skips_rows_without_both_teams():
    assert home_map([{"id": 1, "homeTeam": "Alpha"}]) == {}


def test_neutral_site_games_carry_no_home_edge(league):
    """A neutral game must not push its nominal host up the ratings."""
    games, stats = league
    for game in games:
        game["neutralSite"] = True
    ratings = fit_efficiency(stats, games, season=2025)
    assert ratings.home_field == pytest.approx(0.0, abs=1.0)


# -- integration with the rating book and backtest ---------------------------
def test_rating_book_loads_efficiency_as_a_points_source(league):
    from cfbmeta.ratings import NATIVE_POINT_SOURCES, RatingBook

    games, stats = league
    ratings = fit_efficiency(stats, games, season=2025)
    book = RatingBook(2025)
    assert book.load_efficiency(ratings) == len(ratings.points)
    assert book.rating("Alpha", "efficiency") == pytest.approx(
        ratings.points["alpha"]
    )
    # It arrives already denominated in points, so it must not be z-scored.
    assert "efficiency" in NATIVE_POINT_SOURCES


def test_rating_book_shrugs_off_a_missing_fit():
    from cfbmeta.ratings import RatingBook

    book = RatingBook(2025)
    assert book.load_efficiency(None) == 0
    assert book.load_efficiency(EfficiencyRatings(week=4)) == 0


def test_present_sources_reads_the_rows_not_the_rating_book():
    """The reconstructed basis has components no rating book contains."""
    from cfbmeta.backtest import BacktestRow, present_sources

    rows = [
        BacktestRow(season=2025, week=w, home_team="a", away_team="b",
                    actual_margin=3.0,
                    component_margins={"inseason": 1.0, "efficiency": 2.0},
                    hfa=2.7, adjustments=0.0)
        for w in range(10)
    ]
    assert present_sources(rows) == ["efficiency", "inseason"]


def test_present_sources_drops_a_thinly_covered_component():
    from cfbmeta.backtest import BacktestRow, present_sources

    rows = []
    for i in range(10):
        components = {"inseason": 1.0}
        if i < 2:  # 20% coverage, below the 80% threshold
            components["efficiency"] = 2.0
        rows.append(BacktestRow(season=2025, week=i, home_team="a", away_team="b",
                                actual_margin=3.0, component_margins=components,
                                hfa=2.7, adjustments=0.0))
    assert present_sources(rows) == ["inseason"]


def test_present_sources_on_no_rows():
    from cfbmeta.backtest import present_sources

    assert present_sources([]) == []


def test_fit_weights_defaults_to_what_the_rows_carry():
    """Previously defaulted to ALL_SOURCES and silently fit nothing."""
    from cfbmeta.backtest import BacktestRow, fit_weights

    rows = [
        BacktestRow(season=2025, week=1, home_team="a", away_team="b",
                    actual_margin=float(i % 17) - 8.0,
                    component_margins={"inseason": float(i % 13) - 6.0,
                                       "efficiency": float(i % 11) - 5.0},
                    hfa=2.7, adjustments=0.0)
        for i in range(200)
    ]
    fitted = fit_weights(rows)
    assert set(fitted) == {"inseason", "efficiency"}
    assert sum(fitted.values()) == pytest.approx(1.0, abs=1e-3)


def test_a_weighted_source_absent_from_the_book_does_not_kill_the_slate():
    """Week 1 has no efficiency rating; the projection must still come out.

    Regression: SOURCE_LABELS was indexed directly, so weighting a source the
    rating book had never heard of raised KeyError per game and silently
    emptied the whole slate.
    """
    import datetime as dt

    from cfbmeta.config import Config
    from cfbmeta.model import project_game
    from cfbmeta.ratings import RatingBook

    book = RatingBook(2026)
    book.add("Alpha", "sp_plus", 12.0)
    book.add("Bravo", "sp_plus", 3.0)

    config = Config()
    config.weights.efficiency = 0.20  # weighted, but nothing loaded it

    game = {
        "id": 1, "week": 1, "homeTeam": "Alpha", "awayTeam": "Bravo",
        "startDate": dt.datetime(2026, 9, 5, 19, tzinfo=dt.timezone.utc).isoformat(),
    }
    proj = project_game(game, book, config)
    assert proj is not None
    assert proj.projected_margin is not None
    assert any("Efficiency" in note for note in proj.notes)
