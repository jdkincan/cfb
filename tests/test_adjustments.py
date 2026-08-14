import datetime as dt

import pytest

from cfbmeta.adjustments import (
    build_situational_model,
    haversine_miles,
    parse_start,
)
from cfbmeta.coaching import build_coach_model, normalize_coach
from cfbmeta.hfa import HFAModel, VenueProfile, build_hfa_model, ingest_games
from cfbmeta.ratings import normalize_team

from conftest import SEASON


class TestHFA:
    @pytest.fixture
    def model(self, client):
        return build_hfa_model(client, [SEASON - 1, SEASON])

    def test_neutral_site_gets_no_home_field(self, model):
        result = model.for_game("Ohio State", "Michigan", neutral_site=True)
        assert result["total"] == 0.0

    def test_home_field_is_positive_and_bounded(self, model):
        result = model.for_game("Ohio State", "Michigan", neutral_site=False)
        assert 0.0 <= result["total"] <= model.hfa_max + 1.5

    def test_unknown_team_falls_back_to_league_average(self, model):
        assert model.team_hfa("Some Fake School") == pytest.approx(model.league_hfa)

    def test_shrinkage_pulls_extreme_samples_toward_the_mean(self):
        model = HFAModel(league_hfa=2.5, shrink_games=60.0)
        # Six home games, each beating the rating line by 30 points.
        prof = VenueProfile(team="Fluke State", home_games=6, residual_sum=180.0)
        model.profiles["fluke state"] = prof
        assert prof.raw_hfa == pytest.approx(30.0)
        # Shrunk toward 2.5 and then clamped by hfa_max.
        assert model.team_hfa("Fluke State") <= model.hfa_max

    def test_small_samples_are_ignored(self):
        model = HFAModel(league_hfa=2.5)
        prof = VenueProfile(team="Tiny Sample", home_games=2, residual_sum=60.0)
        model.profiles["tiny sample"] = prof
        assert prof.raw_hfa is None
        assert model.team_hfa("Tiny Sample") == pytest.approx(2.5)

    def test_opponent_strength_is_controlled_for(self):
        """The bug this estimator exists to avoid.

        A team that hosts only weak opponents and visits only strong ones has a
        huge raw home margin and no home-field advantage at all. The residual
        method must report ~0; the old margin-difference method reported ~20.
        """
        model = HFAModel(league_hfa=2.5)
        ratings = {"strong": 20.0, "weak": -20.0, "host": 0.0}
        games = [
            # Hosts the weak team and wins by exactly the rating gap: no edge.
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 20, "awayPoints": 0},
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 30, "awayPoints": 10},
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 20, "awayPoints": 0},
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 27, "awayPoints": 7},
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 21, "awayPoints": 1},
            {"homeTeam": "Host", "awayTeam": "Weak", "homePoints": 24, "awayPoints": 4},
        ]
        assert ingest_games(model, games, ratings) == 6
        assert model.profiles["host"].raw_hfa == pytest.approx(0.0, abs=0.01)

    def test_games_without_ratings_are_skipped(self):
        model = HFAModel()
        games = [{"homeTeam": "A", "awayTeam": "B", "homePoints": 30, "awayPoints": 10}]
        # No ratings supplied: the game cannot be opponent-controlled.
        assert ingest_games(model, games, {}) == 0

    def test_altitude_helps_wyoming_against_a_sea_level_visitor(self, model):
        laramie = model.for_game("Wyoming", "Florida", neutral_site=False)
        assert laramie["elevation"] > 0.8

    def test_altitude_edge_shrinks_between_two_high_venues(self, model):
        vs_air_force = model.elevation_edge("Colorado", "Air Force")
        vs_florida = model.elevation_edge("Colorado", "Florida")
        assert vs_air_force < vs_florida

    def test_low_altitude_venue_gets_no_elevation_credit(self, model):
        assert model.elevation_edge("Florida", "Wyoming") == 0.0

    def test_ingest_skips_neutral_and_unplayed_games(self):
        model = HFAModel()
        ratings = {"a": 5.0, "b": 0.0}
        games = [
            {"homeTeam": "A", "awayTeam": "B", "homePoints": 30, "awayPoints": 10,
             "neutralSite": True},
            {"homeTeam": "A", "awayTeam": "B", "homePoints": None, "awayPoints": None,
             "neutralSite": False},
            {"homeTeam": "A", "awayTeam": "B", "homePoints": 30, "awayPoints": 10,
             "neutralSite": False},
        ]
        assert ingest_games(model, games, ratings) == 1

    def test_league_average_is_recentred_from_data(self, model):
        assert 0.0 < model.league_hfa < 6.0


class TestCoaching:
    @pytest.fixture
    def model(self, client):
        return build_coach_model(client, SEASON, lookback_years=4)

    def test_coaches_are_mapped_to_teams(self, model):
        assert model.coach_for("Ohio State") == "Ryan Day"
        assert model.coach_for("Georgia") == "Kirby Smart"

    def test_scores_are_capped(self, model):
        for team in ("Ohio State", "UMass", "Georgia", "Wyoming"):
            assert abs(model.team_score(team)) <= model.cap + 1e-9

    def test_first_year_coach_is_penalized(self, model):
        assert model.first_year_teams.get("umass") is True
        assert model.team_score("UMass") < 0

    def test_returning_coach_is_not_flagged_first_year(self, model):
        assert model.first_year_teams.get("ohio state") is False

    def test_game_adjustment_is_antisymmetric(self, model):
        forward = model.for_game("Ohio State", "Michigan")["total"]
        reverse = model.for_game("Michigan", "Ohio State")["total"]
        assert forward == pytest.approx(-reverse)

    def test_unknown_team_scores_zero(self, model):
        assert model.team_score("Fake U") == 0.0

    def test_talent_relationship_is_positive(self, model):
        # More talent should predict a better SP+ rating.
        assert model.talent_slope > 0

    def test_missing_coach_data_degrades_quietly(self, client):
        from conftest import FakeClient

        model = build_coach_model(FakeClient(fail={"coaches"}), SEASON, lookback_years=2)
        assert model.for_game("Ohio State", "Michigan")["total"] == 0.0

    def test_normalize_coach_handles_spacing(self):
        assert normalize_coach("  Ryan   Day ") == "ryan day"


class TestSituational:
    @pytest.fixture
    def model(self, client):
        return build_situational_model(client, SEASON)

    def test_haversine_is_sane(self):
        # Columbus to Eugene is roughly 2,000 miles.
        miles = haversine_miles(40.0017, -83.0197, 44.0583, -123.0681)
        assert 1900 < miles < 2200

    def test_haversine_zero_for_same_point(self):
        assert haversine_miles(40.0, -83.0, 40.0, -83.0) == pytest.approx(0.0)

    def test_parse_start_handles_formats(self):
        assert parse_start("2025-10-04T19:00:00.000Z").year == 2025
        assert parse_start("2025-10-04T19:00:00+00:00").hour == 19
        assert parse_start("2025-10-04").day == 4
        assert parse_start(None) is None
        assert parse_start("nonsense") is None

    def test_parse_start_assumes_utc_when_naive(self):
        assert parse_start("2025-10-04").tzinfo is not None

    def test_long_travel_is_penalized(self, model):
        kickoff = dt.datetime(SEASON, 10, 4, 19, tzinfo=dt.timezone.utc)
        # UMass travelling to Hawai'i is about as far as college football gets.
        result = model.for_game("Hawai'i", "UMass", kickoff)
        assert result["travel"]["miles"] > 4000
        assert result["travel"]["total"] > 0

    def test_short_trip_is_not_penalized(self, model):
        kickoff = dt.datetime(SEASON, 10, 4, 19, tzinfo=dt.timezone.utc)
        result = model.for_game("Colorado", "Air Force", kickoff)
        assert result["travel"]["total"] == pytest.approx(0.0, abs=0.05)

    def test_travel_is_capped(self, model):
        kickoff = dt.datetime(SEASON, 10, 4, 19, tzinfo=dt.timezone.utc)
        result = model.for_game("Hawai'i", "UMass", kickoff)
        assert result["travel"]["total"] <= model.travel_cap

    def test_rest_advantage_favours_the_rested_team(self):
        from cfbmeta.adjustments import SituationalModel

        model = SituationalModel()
        kickoff = dt.datetime(2025, 10, 11, 19, tzinfo=dt.timezone.utc)
        model.schedule[normalize_team("Home U")] = [dt.datetime(2025, 9, 27, 19, tzinfo=dt.timezone.utc)]
        model.schedule[normalize_team("Away U")] = [dt.datetime(2025, 10, 4, 19, tzinfo=dt.timezone.utc)]
        result = model.rest_adjustment("Home U", "Away U", kickoff)
        assert result["home_days"] == pytest.approx(14.0)
        assert result["away_days"] == pytest.approx(7.0)
        assert result["total"] > 0

    def test_short_week_penalizes_the_home_side(self):
        from cfbmeta.adjustments import SituationalModel

        model = SituationalModel()
        kickoff = dt.datetime(2025, 10, 9, 19, tzinfo=dt.timezone.utc)
        model.schedule[normalize_team("Home U")] = [dt.datetime(2025, 10, 5, 19, tzinfo=dt.timezone.utc)]
        model.schedule[normalize_team("Away U")] = [dt.datetime(2025, 9, 28, 19, tzinfo=dt.timezone.utc)]
        result = model.rest_adjustment("Home U", "Away U", kickoff)
        assert result["total"] < 0

    def test_rest_is_capped(self):
        from cfbmeta.adjustments import SituationalModel

        model = SituationalModel(rest_cap=2.0)
        kickoff = dt.datetime(2025, 11, 1, 19, tzinfo=dt.timezone.utc)
        model.schedule[normalize_team("Home U")] = [dt.datetime(2025, 8, 1, 19, tzinfo=dt.timezone.utc)]
        model.schedule[normalize_team("Away U")] = [dt.datetime(2025, 10, 29, 19, tzinfo=dt.timezone.utc)]
        assert model.rest_adjustment("Home U", "Away U", kickoff)["total"] <= 2.0

    def test_missing_schedule_yields_no_adjustment(self):
        from cfbmeta.adjustments import SituationalModel

        model = SituationalModel()
        kickoff = dt.datetime(2025, 10, 11, 19, tzinfo=dt.timezone.utc)
        assert model.rest_adjustment("Nobody", "Nobody Else", kickoff)["total"] == 0.0
