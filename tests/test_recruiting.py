"""Talent rebuilt from recruiting classes, and the FBS-only screen."""

import pytest

from cfbmeta.config import Config
from cfbmeta.model import project_slate
from cfbmeta.ratings import normalize_team
from cfbmeta.sources.recruiting import (
    CLASS_WEIGHTS,
    build_recruiting_profiles,
    blue_chip_table,
    talent_rows,
)

SEASON = 2026


class FakeRecruitingClient:
    """Two blue-blood-ish teams and one that signs nobody good."""

    ROSTER = {
        "Blue Blood": (5, 15, 4),   # (5-star, 4-star, 3-star) per class
        "Solid State": (0, 6, 16),
        "Directional": (0, 0, 22),
    }
    POINTS = {"Blue Blood": 300.0, "Solid State": 190.0, "Directional": 120.0}

    def __init__(self, missing_players=False):
        self.missing_players = missing_players

    def get(self, endpoint, **params):
        year = params.get("year")
        if endpoint == "recruiting_teams":
            return [{"year": year, "team": t, "rank": i + 1, "points": p}
                    for i, (t, p) in enumerate(self.POINTS.items())]
        if endpoint == "recruiting_players":
            if self.missing_players:
                return []
            rows = []
            for team, (five, four, three) in self.ROSTER.items():
                for stars, n in ((5, five), (4, four), (3, three)):
                    rows += [{"year": year, "committedTo": team, "stars": stars,
                              "name": f"{team} {stars}star {i}"} for i in range(n)]
            return rows
        raise AssertionError(endpoint)


@pytest.fixture
def profiles():
    return build_recruiting_profiles(FakeRecruitingClient(), SEASON)


class TestRecruitingProfiles:
    def test_every_team_is_profiled(self, profiles):
        assert set(profiles) == {normalize_team(t) for t in FakeRecruitingClient.ROSTER}

    def test_four_classes_are_aggregated(self, profiles):
        assert profiles[normalize_team("Blue Blood")].classes_seen == 4

    def test_class_weights_are_applied(self, profiles):
        # Points are identical each year, so the total is points * sum(weights).
        expected = 300.0 * sum(CLASS_WEIGHTS.values())
        assert profiles[normalize_team("Blue Blood")].points == pytest.approx(expected)

    def test_recent_classes_count_for_more_than_the_incoming_one(self):
        # Freshmen play less than upperclassmen; the weighting must reflect it.
        assert CLASS_WEIGHTS[0] < CLASS_WEIGHTS[1]
        assert CLASS_WEIGHTS[1] >= CLASS_WEIGHTS[3]

    def test_talent_ordering_matches_recruiting_strength(self, profiles):
        pts = {t: p.points for t, p in profiles.items()}
        assert pts[normalize_team("Blue Blood")] > pts[normalize_team("Solid State")]
        assert pts[normalize_team("Solid State")] > pts[normalize_team("Directional")]


class TestBlueChipRatio:
    def test_ratio_counts_four_and_five_stars(self, profiles):
        blue = profiles[normalize_team("Blue Blood")]
        # 20 of 24 signees per class are blue chips.
        assert blue.blue_chip_ratio == pytest.approx(20 / 24)

    def test_a_team_with_no_blue_chips_scores_zero(self, profiles):
        assert profiles[normalize_team("Directional")].blue_chip_ratio == 0.0

    def test_small_samples_return_none(self):
        from cfbmeta.sources.recruiting import TeamRecruiting

        assert TeamRecruiting(team="Tiny", signees=5, blue_chips=3).blue_chip_ratio is None

    def test_table_is_sorted_best_first(self, profiles):
        table = blue_chip_table(profiles)
        assert [r[0] for r in table][:2] == ["Blue Blood", "Solid State"]
        assert table[0][1] >= table[-1][1]


class TestTalentRows:
    def test_shaped_like_the_talent_endpoint(self, profiles):
        rows = talent_rows(profiles)
        assert all(set(r) == {"team", "talent"} for r in rows)
        assert len(rows) == 3

    def test_missing_player_data_still_yields_talent(self):
        # Team points alone are enough for the talent source; only the
        # blue-chip ratio needs the player rows.
        profiles = build_recruiting_profiles(FakeRecruitingClient(missing_players=True), SEASON)
        assert len(talent_rows(profiles)) == 3
        assert all(p.blue_chip_ratio is None for p in profiles.values())


class TestFBSScreen:
    def _games(self):
        return [
            {"id": 1, "week": 1, "season": SEASON, "homeTeam": "Ohio State",
             "awayTeam": "Michigan", "startDate": "2026-09-05T19:00:00.000Z"},
            {"id": 2, "week": 1, "season": SEASON, "homeTeam": "Ohio State",
             "awayTeam": "Youngstown State", "startDate": "2026-09-05T19:00:00.000Z"},
        ]

    def test_non_fbs_opponents_are_screened_out(self, book, config):
        fbs = {normalize_team("Ohio State"), normalize_team("Michigan")}
        projections = project_slate(self._games(), book, config, fbs_teams=fbs)
        assert [p.away_team for p in projections] == ["Michigan"]

    def test_screen_can_be_disabled(self, book, config):
        config.fbs_only = False
        fbs = {normalize_team("Ohio State"), normalize_team("Michigan")}
        projections = project_slate(self._games(), book, config, fbs_teams=fbs)
        # Youngstown State still has no ratings, so it drops for that reason.
        assert len(projections) == 1

    def test_no_list_means_no_screen(self, book, config):
        projections = project_slate(self._games(), book, config, fbs_teams=None)
        assert len(projections) == 1
