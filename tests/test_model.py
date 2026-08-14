import pytest

from cfbmeta.adjustments import build_situational_model
from cfbmeta.coaching import build_coach_model
from cfbmeta.config import Config, Weights
from cfbmeta.hfa import build_hfa_model
from cfbmeta.model import (
    ComponentMargin,
    blend,
    component_margins,
    detect_defense_sign,
    effective_weights,
    project_game,
    project_slate,
)
from cfbmeta.sources.market import build_market_map, consensus_line

from conftest import SEASON


@pytest.fixture
def models(client):
    return {
        "hfa_model": build_hfa_model(client, [SEASON - 1, SEASON]),
        "coach_model": build_coach_model(client, SEASON, lookback_years=4),
        "situational": build_situational_model(client, SEASON),
    }


@pytest.fixture
def markets(client):
    return build_market_map(client.lines(SEASON, week=6))


class TestWeights:
    def test_normalized_weights_sum_to_one(self):
        w = Weights().normalized()
        assert sum(w.values()) == pytest.approx(1.0)

    def test_market_weight_is_carved_out_not_added(self):
        w = Weights(market=0.30).normalized()
        assert w["market"] == pytest.approx(0.30)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_zero_model_weight_is_rejected(self):
        with pytest.raises(ValueError):
            Weights(sp_plus=0, fpi=0, elo=0, srs=0, talent=0).normalized()

    def test_early_season_downweights_in_season_sources(self, config):
        week1 = effective_weights(config, 1)
        week10 = effective_weights(config, 10)
        assert week1["elo"] < week10["elo"]
        assert week1["srs"] < week10["srs"]
        assert week1["sp_plus"] > week10["sp_plus"]

    def test_weights_still_sum_to_one_after_shifting(self, config):
        for week in range(1, 15):
            assert sum(effective_weights(config, week).values()) == pytest.approx(1.0)

    def test_late_season_weights_are_unshifted(self, config):
        assert effective_weights(config, 12) == pytest.approx(config.weights.normalized())


class TestBlend:
    def test_weighted_mean(self):
        comps = [
            ComponentMargin("sp_plus", "SP+", 10.0, 0.5),
            ComponentMargin("fpi", "FPI", 20.0, 0.5),
        ]
        assert blend(comps) == pytest.approx(15.0)

    def test_renormalizes_over_available_sources(self):
        # Weights sum to 0.25 but the result should still be a proper average.
        comps = [
            ComponentMargin("sp_plus", "SP+", 10.0, 0.2),
            ComponentMargin("fpi", "FPI", 20.0, 0.05),
        ]
        assert blend(comps) == pytest.approx((10 * 0.2 + 20 * 0.05) / 0.25)

    def test_empty_blend_is_zero(self):
        assert blend([]) == 0.0

    def test_missing_source_drops_out(self, book, config):
        weights = config.weights.normalized()
        comps = component_margins(book, "Ohio State", "Not A Team", weights)
        assert comps == []

    def test_component_margins_are_neutral_field(self, book, config):
        weights = config.weights.normalized()
        comps = component_margins(book, "Ohio State", "Michigan", weights)
        sp = next(c for c in comps if c.source == "sp_plus")
        assert sp.margin == pytest.approx(28.4 - 16.2)

    def test_market_component_included_only_when_weighted(self, book, config):
        weights = config.weights.normalized()
        assert not any(
            c.source == "market"
            for c in component_margins(book, "Ohio State", "Michigan", weights, 7.0)
        )
        config.weights.market = 0.3
        weighted = config.weights.normalized()
        assert any(
            c.source == "market"
            for c in component_margins(book, "Ohio State", "Michigan", weighted, 7.0)
        )


class TestDefenseSign:
    def test_detects_lower_is_better(self, book):
        # The fixture builds defense as 25 - rating/2, so a lower defense
        # number goes with a better team.
        assert detect_defense_sign(book) == -1.0

    def test_detects_higher_is_better_when_the_data_flips(self, book):
        # If CFBD ever quotes defense so that higher = better, the sign must
        # follow the data rather than a hardcoded assumption.
        for entry in book.teams.values():
            if "sp_defense" in entry.meta:
                entry.meta["sp_defense"] = -entry.meta["sp_defense"]
        assert detect_defense_sign(book) == 1.0

    def test_falls_back_when_there_is_too_little_data(self):
        from cfbmeta.ratings import RatingBook

        assert detect_defense_sign(RatingBook(SEASON)) == -1.0


class TestProjectGame:
    def test_projection_has_all_the_pieces(self, book, config, models, week6_games, markets):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, market=markets.get(game["id"]), **models)

        assert proj.home_team == "Ohio State"
        assert proj.away_team == "Michigan"
        assert len(proj.components) >= 4
        assert proj.projected_margin != 0
        assert proj.hfa["total"] > 0
        assert 0 < proj.home_win_probability < 1
        assert proj.spread_bet is not None

    def test_better_team_at_home_is_favored(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, **models)
        assert proj.projected_margin > 0
        assert proj.favorite == "Ohio State"
        assert proj.home_win_probability > 0.5

    def test_home_field_is_added_once_not_per_source(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, **models)
        expected = proj.neutral_margin + proj.hfa["total"] + proj.adjustment_total
        assert proj.projected_margin == pytest.approx(expected, abs=0.01)

    def test_neutral_site_drops_home_field(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["neutralSite"])
        proj = project_game(game, book, config, **models)
        assert proj.hfa["total"] == 0.0
        assert "vs" in proj.matchup

    def test_edge_is_projection_minus_market(self, book, config, models, week6_games, markets):
        game = next(g for g in week6_games if g["homeTeam"] == "Texas")
        proj = project_game(game, book, config, market=markets.get(game["id"]), **models)
        assert proj.market_spread is not None
        assert proj.edge == pytest.approx(
            proj.projected_margin - (-proj.market_spread), abs=0.01
        )

    def test_no_market_means_no_play(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, **models)
        assert proj.market_spread is None
        assert proj.spread_bet.is_play is False
        assert any("market" in n.lower() for n in proj.notes)

    def test_adjustment_total_is_capped(self, book, config, models, week6_games):
        config.total_adj_cap = 1.0
        for game in week6_games:
            proj = project_game(game, book, config, **models)
            assert abs(proj.adjustment_total) <= 1.0 + 1e-9

    def test_projected_line_reads_like_a_book_quote(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, **models)
        assert proj.projected_line.startswith("Ohio State -")

    def test_totals_are_consistent_with_the_margin(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Ohio State")
        proj = project_game(game, book, config, **models)
        assert proj.projected_total is not None
        assert proj.projected_home_points - proj.projected_away_points == pytest.approx(
            proj.projected_margin, abs=0.15
        )
        assert proj.projected_home_points + proj.projected_away_points == pytest.approx(
            proj.projected_total, abs=0.15
        )

    def test_wyoming_altitude_shows_up_in_the_ledger(self, book, config, models, week6_games):
        game = next(g for g in week6_games if g["homeTeam"] == "Wyoming")
        proj = project_game(game, book, config, **models)
        assert proj.hfa["elevation"] > 0.5

    def test_unrated_opponent_produces_a_note(self, book, config, models):
        game = {
            "id": 1, "week": 6, "season": SEASON,
            "homeTeam": "Ohio State", "awayTeam": "Nowhere State",
            "startDate": "2025-10-04T19:00:00.000Z", "neutralSite": False,
        }
        proj = project_game(game, book, config, **models)
        assert proj.components == []
        assert proj.notes


class TestProjectSlate:
    def test_every_rated_game_is_projected(self, book, config, models, week6_games, markets):
        projections = project_slate(week6_games, book, config, markets=markets, **models)
        assert len(projections) == len(week6_games)

    def test_sorted_by_edge_descending(self, book, config, models, week6_games, markets):
        projections = project_slate(week6_games, book, config, markets=markets, **models)
        keys = [p.sort_key() for p in projections]
        assert keys == sorted(keys, reverse=True)

    def test_games_with_unrated_teams_are_skipped(self, book, config, models, week6_games):
        games = week6_games + [
            {"id": 99, "week": 6, "season": SEASON, "homeTeam": "Ohio State",
             "awayTeam": "Directional FCS", "startDate": "2025-10-04T19:00:00.000Z"}
        ]
        projections = project_slate(games, book, config, **models)
        assert all(p.away_team != "Directional FCS" for p in projections)

    def test_deliberate_market_disagreement_becomes_a_play(
        self, book, config, models, week6_games, markets
    ):
        # The fixture prices Texas 4 points off our number and Oregon 6 the
        # other way; both should clear the play threshold.
        projections = project_slate(week6_games, book, config, markets=markets, **models)
        by_home = {p.home_team: p for p in projections}
        assert by_home["Texas"].spread_bet.is_play
        assert by_home["Oregon"].spread_bet.is_play
        assert by_home["Texas"].spread_bet.side != by_home["Oregon"].spread_bet.side

    def test_units_never_exceed_the_cap(self, book, config, models, week6_games, markets):
        projections = project_slate(week6_games, book, config, markets=markets, **models)
        for proj in projections:
            assert proj.spread_bet.units <= config.max_units_per_play


class TestMarketConsolidation:
    def test_median_spread_across_books(self):
        row = {
            "id": 1, "homeTeam": "A", "awayTeam": "B",
            "lines": [
                {"provider": "DraftKings", "spread": -7.0, "overUnder": 52.5},
                {"provider": "Bovada", "spread": -7.5, "overUnder": 53.0},
                {"provider": "consensus", "spread": -20.0},  # obvious outlier
            ],
        }
        line = consensus_line(row)
        assert line["spread"] == pytest.approx(-7.5)
        assert line["book_count"] == 3

    def test_reports_spread_range(self):
        row = {"id": 1, "lines": [
            {"provider": "a", "spread": -7.0}, {"provider": "b", "spread": -8.0}]}
        assert consensus_line(row)["spread_range"] == (-8.0, -7.0)

    def test_row_without_lines_is_dropped(self):
        assert consensus_line({"id": 1, "lines": []}) is None

    def test_map_is_keyed_by_id_and_team_pair(self, client):
        markets = build_market_map(client.lines(SEASON, week=6))
        assert any(isinstance(k, int) for k in markets)
        assert any(isinstance(k, tuple) for k in markets)
