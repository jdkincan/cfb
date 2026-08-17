"""Snapshot archive, SP+ publication check, and the team spotlight."""

import csv
import datetime as dt
import json

import pytest

from cfbmeta.archive import grade, list_snapshots, projection_rows, snapshot, week_dir
from cfbmeta.model import project_slate
from cfbmeta.ratings import build_rating_book
from cfbmeta.sources.spplus_check import CheckResult, check, parse_claims
from cfbmeta.spotlight import build_spotlight, rating_momentum

from conftest import SEASON, FakeClient

ARTICLE = """
<p>Oklahoma State (up 22.0 points, 39th overall) rebounds. And a major
production boost on both sides of the ball have the Cowboys likely to rebound
pretty significantly. Virginia Tech (up 18.7 points, 36th overall) too.
North Texas (down 25.4 points, 105th overall) falls hard, and at least,
Indiana (down 6.6 points, fifth overall) slips.</p>
"""


class TestArticleParsing:
    def test_extracts_each_claim(self):
        claims = parse_claims(ARTICLE)
        assert [c.team for c in claims] == [
            "Oklahoma State", "Virginia Tech", "North Texas", "Indiana"
        ]

    def test_direction_is_signed(self):
        by = {c.team: c.delta for c in parse_claims(ARTICLE)}
        assert by["Oklahoma State"] == pytest.approx(22.0)
        assert by["North Texas"] == pytest.approx(-25.4)

    def test_word_ordinals_are_understood(self):
        assert next(c for c in parse_claims(ARTICLE) if c.team == "Indiana").rank == 5

    def test_numeric_ordinals_are_understood(self):
        assert next(c for c in parse_claims(ARTICLE) if c.team == "North Texas").rank == 105

    def test_prose_is_not_captured_as_a_team_name(self):
        """The bug that cost half the checks: IGNORECASE let [A-Z] match
        lowercase, so a whole sentence became the 'team name'."""
        for claim in parse_claims(ARTICLE):
            assert len(claim.team.split()) <= 4
            assert claim.team[0].isupper()

    def test_empty_input_is_safe(self):
        assert parse_claims("") == []
        assert parse_claims(None) == []


class TestPublicationCheck:
    def _session(self, story=ARTICLE):
        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"headlines": [{"headline": "SP+", "published": "2026-08-13T00:00:00Z",
                                       "story": story}]}

        class Session:
            def get(self, *a, **k):
                return Response()

        return Session()

    def test_matching_ratings_pass(self):
        current = {"oklahoma state": 6.9, "north texas": -11.6}
        previous = {"oklahoma state": -15.1, "north texas": 13.8}
        result = check("1", current, previous, session=self._session())
        assert result.ok
        assert result.matched == 2

    def test_a_disagreement_is_reported(self):
        current = {"oklahoma state": 99.0}
        previous = {"oklahoma state": -15.1}
        result = check("1", current, previous, session=self._session())
        assert not result.ok
        assert result.mismatched[0][0] == "Oklahoma State"
        assert "DISAGREES" in result.describe()

    def test_rounding_tolerance_is_allowed(self):
        # Article rounds to a decimal; 22.04 must still match "up 22.0".
        result = check("1", {"oklahoma state": 6.94}, {"oklahoma state": -15.1},
                       session=self._session())
        assert result.ok

    def test_unknown_teams_are_skipped_not_failed(self):
        result = check("1", {}, {}, session=self._session())
        assert result.checked == 0
        assert not result.ok

    def test_a_network_failure_never_raises(self):
        class Broken:
            def get(self, *a, **k):
                raise RuntimeError("down")

        result = check("1", {}, {}, session=Broken())
        assert result.error
        assert "unavailable" in result.describe()


@pytest.fixture
def week_projections(client, book, config):
    games = [g for g in client.games(SEASON) if g["week"] == 6]
    return games, project_slate(games, book, config)


class TestSnapshot:
    def test_writes_every_file(self, tmp_path, book, config, week_projections):
        _, projections = week_projections
        directory = snapshot(SEASON, 6, book=book, projections=projections,
                             markets={1: {"spread": -7.0, "total": 52.5}},
                             config=config, root=tmp_path)
        for name in ("ratings.csv", "projections.csv", "lines.csv", "meta.json"):
            assert (directory / name).exists(), name

    def test_ratings_capture_every_source(self, tmp_path, book, config):
        directory = snapshot(SEASON, 6, book=book, config=config, root=tmp_path)
        with (directory / "ratings.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        sources = {r["source"] for r in rows}
        assert {"sp_plus", "fpi", "elo", "srs", "talent"} <= sources

    def test_meta_records_provenance_and_config(self, tmp_path, book, config):
        directory = snapshot(SEASON, 6, book=book, config=config, root=tmp_path)
        meta = json.loads((directory / "meta.json").read_text())
        assert meta["season"] == SEASON and meta["week"] == 6
        assert meta["usable_sources"]
        assert meta["config"]["sigma_margin"] == config.sigma_margin
        assert "captured_at" in meta

    def test_projection_rows_are_gradeable(self, week_projections):
        _, projections = week_projections
        rows = projection_rows(projections)
        assert rows
        for row in rows:
            for column in ("home_points", "away_points", "actual_margin", "bet_result"):
                assert column in row

    def test_snapshots_are_discoverable(self, tmp_path, book, config):
        snapshot(SEASON, 5, book=book, config=config, root=tmp_path)
        snapshot(SEASON, 6, book=book, config=config, root=tmp_path)
        found = list_snapshots(SEASON, tmp_path)
        assert [p.name for p in found] == ["week-05", "week-06"]

    def test_week_directories_are_zero_padded(self, tmp_path):
        assert week_dir(2026, 3, tmp_path).name == "week-03"


class TestGrading:
    def test_results_are_written_back(self, tmp_path, book, config, week_projections):
        games, projections = week_projections
        snapshot(SEASON, 6, book=book, projections=projections, config=config, root=tmp_path)

        played = [dict(g, homePoints=31, awayPoints=17) for g in games]
        assert grade(SEASON, 6, played, root=tmp_path) == len(projections)

        path = week_dir(SEASON, 6, tmp_path) / "projections.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert all(row["actual_margin"] == "14.0" for row in rows)

    def test_bet_results_are_graded_by_side(self, tmp_path, book, config, week_projections):
        games, projections = week_projections
        snapshot(SEASON, 6, book=book, projections=projections, config=config, root=tmp_path)
        grade(SEASON, 6, [dict(g, homePoints=31, awayPoints=17) for g in games], root=tmp_path)

        with (week_dir(SEASON, 6, tmp_path) / "projections.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        graded = [r for r in rows if r["bet_result"]]
        assert all(r["bet_result"] in ("win", "loss", "push") for r in graded)

    def test_grading_a_missing_snapshot_is_safe(self, tmp_path):
        assert grade(2026, 9, [], root=tmp_path) == 0


class TestSpotlight:
    def test_finds_the_followed_team(self, book, config, week_projections):
        _, projections = week_projections
        spot = build_spotlight(projections, "Ohio State", book=book)
        assert spot is not None
        assert "Ohio State" in (spot.home.team, spot.away.team)

    def test_returns_none_when_the_team_is_idle(self, book, week_projections):
        _, projections = week_projections
        assert build_spotlight(projections, "Nobody State", book=book) is None

    def test_disabled_by_empty_team(self, book, week_projections):
        _, projections = week_projections
        assert build_spotlight(projections, "", book=book) is None

    def test_falls_back_to_the_unfiltered_schedule(self, book, config, week_projections):
        """A team playing an FCS opponent is off the slate but still followed."""
        games, projections = week_projections
        dropped = [p for p in projections if p.home_team != "Wyoming"]
        raw = [g for g in games if g["homeTeam"] == "Wyoming"]

        from cfbmeta.model import project_game

        spot = build_spotlight(
            dropped, "Wyoming", book=book,
            all_games=raw, project_one=lambda g: project_game(g, book, config),
        )
        assert spot is not None
        assert "bettable slate" in spot.note

    def test_momentum_reads_the_archive(self, tmp_path, book, config):
        snapshot(SEASON, 4, book=book, config=config, root=tmp_path)
        snapshot(SEASON, 5, book=book, config=config, root=tmp_path)
        series = rating_momentum("Ohio State", SEASON, archive_root=tmp_path)
        assert [w for w, _ in series] == [4, 5]

    def test_momentum_is_empty_without_history(self, tmp_path):
        assert rating_momentum("Ohio State", SEASON, archive_root=tmp_path) == []

    def test_momentum_deltas(self, book, config, week_projections):
        from cfbmeta.spotlight import TeamDetail

        detail = TeamDetail(team="X", momentum=[(1, 10.0), (2, 12.0), (3, 15.0)])
        assert detail.momentum_delta == pytest.approx(5.0)
        assert detail.momentum_recent == pytest.approx(3.0)

    def test_single_week_has_no_delta(self):
        from cfbmeta.spotlight import TeamDetail

        assert TeamDetail(team="X", momentum=[(1, 10.0)]).momentum_delta is None
