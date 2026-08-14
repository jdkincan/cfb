"""Roster-based talent, and the FBS-only screen.

The talent composite must reflect who is *on the team now*, not who signed.
These tests pin the transfer behaviour that distinction exists for.
"""

import pytest

from cfbmeta.model import project_slate
from cfbmeta.ratings import normalize_team
from cfbmeta.sources.recruiting import (
    TOP_N,
    WALKON_RATING,
    TeamRoster,
    blue_chip_table,
    build_roster_talent,
    talent_rows,
)

SEASON = 2026


class FakeRosterClient:
    """Three teams, and one star who transfers between them."""

    # athleteId -> (rating, stars)
    RECRUITS = {
        **{f"blue{i}": (0.95, 5 if i < 4 else 4) for i in range(30)},
        **{f"mid{i}": (0.86, 3) for i in range(30)},
        **{f"low{i}": (0.80, 3) for i in range(30)},
        "star": (0.99, 5),
    }

    def __init__(self, star_plays_for="Blue Blood", walkons=0):
        self.star_plays_for = star_plays_for
        self.walkons = walkons

    def get(self, endpoint, **params):
        if endpoint == "recruiting_players":
            year = params["year"]
            # Spread the pool across classes so the lookback is exercised.
            return [
                {"year": year, "athleteId": aid, "rating": r, "stars": s,
                 "committedTo": "Somewhere"}
                for aid, (r, s) in self.RECRUITS.items()
                if hash(aid) % 8 == year % 8
            ]
        if endpoint == "roster":
            rows = []
            for team, prefix in (("Blue Blood", "blue"), ("Mid Major", "mid"),
                                 ("Directional", "low")):
                rows += [{"id": f"{prefix}{i}", "team": team} for i in range(30)]
                rows += [{"id": f"walkon-{team}-{i}", "team": team}
                         for i in range(self.walkons)]
            rows.append({"id": "star", "team": self.star_plays_for})
            return rows
        raise AssertionError(endpoint)


@pytest.fixture
def teams():
    return build_roster_talent(FakeRosterClient(), SEASON)


class TestRosterTalent:
    def test_every_team_is_scored(self, teams):
        assert set(teams) == {
            normalize_team(t) for t in ("Blue Blood", "Mid Major", "Directional")
        }

    def test_ordering_follows_roster_quality(self, teams):
        blue = teams[normalize_team("Blue Blood")].talent
        mid = teams[normalize_team("Mid Major")].talent
        low = teams[normalize_team("Directional")].talent
        assert blue > mid > low

    def test_roster_size_is_counted_separately_from_matches(self, teams):
        entry = teams[normalize_team("Blue Blood")]
        assert entry.roster_size >= entry.matched

    def test_talent_uses_only_the_top_slots(self):
        entry = TeamRoster(team="Deep", ratings=[0.99] * 100, roster_size=100)
        # 100 elite players cannot score more than TOP_N of them.
        assert entry.talent == pytest.approx(0.99 * TOP_N)

    def test_unmatched_roster_spots_are_imputed_not_dropped(self):
        """Dropping them would flatter teams with poor match coverage."""
        sparse = TeamRoster(team="Sparse", ratings=[0.95] * 10, roster_size=85)
        # 10 matched + 30 imputed walk-ons, not 10 players' worth.
        assert sparse.talent == pytest.approx(0.95 * 10 + WALKON_RATING * 30)

    def test_a_short_roster_is_not_padded_past_its_size(self):
        tiny = TeamRoster(team="Tiny", ratings=[0.95] * 5, roster_size=12)
        assert tiny.talent == pytest.approx(0.95 * 5 + WALKON_RATING * 7)

    def test_walkons_dilute_a_thin_roster(self):
        """A team of 30 rated players plus walk-ons scores below 40 rated."""
        loaded = TeamRoster(team="Loaded", ratings=[0.90] * 40, roster_size=85)
        thin = TeamRoster(team="Thin", ratings=[0.90] * 30, roster_size=85)
        assert loaded.talent > thin.talent


class TestTransfersAreHandled:
    """The reason this is roster-based rather than signing-class-based."""

    def test_a_transfer_counts_for_his_new_team(self):
        before = build_roster_talent(FakeRosterClient(star_plays_for="Directional"), SEASON)
        after = build_roster_talent(FakeRosterClient(star_plays_for="Blue Blood"), SEASON)
        directional = normalize_team("Directional")
        # The star leaving Directional must reduce Directional's talent.
        assert after[directional].talent < before[directional].talent

    def test_the_transfer_is_not_double_counted(self):
        teams = build_roster_talent(FakeRosterClient(star_plays_for="Mid Major"), SEASON)
        appearances = sum(
            1 for t in teams.values() if any(r == 0.99 for r in t.ratings)
        )
        assert appearances == 1


class TestBlueChipRatio:
    def test_measures_the_rated_roster(self, teams):
        blue = teams[normalize_team("Blue Blood")]
        assert blue.blue_chip_ratio == pytest.approx(1.0)

    def test_a_three_star_roster_scores_zero(self, teams):
        assert teams[normalize_team("Directional")].blue_chip_ratio == 0.0

    def test_small_samples_return_none(self):
        assert TeamRoster(team="Tiny", ratings=[0.9] * 5, blue_chips=3).blue_chip_ratio is None

    def test_table_is_sorted_best_first(self, teams):
        table = blue_chip_table(teams)
        assert table[0][1] >= table[-1][1]


class TestTalentRows:
    def test_shaped_like_the_talent_endpoint(self, teams):
        rows = talent_rows(teams)
        assert all(set(r) == {"team", "talent"} for r in rows)
        assert len(rows) == 3

    def test_tiny_rosters_are_excluded(self):
        class OnePlayer(FakeRosterClient):
            def get(self, endpoint, **params):
                if endpoint == "roster":
                    return [{"id": "star", "team": "Ghost"}]
                return super().get(endpoint, **params)

        assert talent_rows(build_roster_talent(OnePlayer(), SEASON)) == []

    def test_missing_roster_yields_nothing(self):
        class NoRoster(FakeRosterClient):
            def get(self, endpoint, **params):
                if endpoint == "roster":
                    raise RuntimeError("unavailable")
                return super().get(endpoint, **params)

        assert build_roster_talent(NoRoster(), SEASON) == {}


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
