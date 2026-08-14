"""ESPN FPI fallback.

The endpoint is undocumented and could not be reached from the machine this was
written on, so the parser is deliberately tolerant and these tests pin that
tolerance: several plausible payload shapes must all yield the same rows, and
anything ambiguous must yield nothing rather than a wrong number.
"""

import logging

import pytest

from cfbmeta.ratings import build_rating_book, normalize_team
from cfbmeta.sources.espn import extract_fpi_rows, fetch_fpi

from conftest import TEAMS, FakeClient
from test_preseason import PreseasonClient

SEASON = 2026


# -- payload shapes ESPN has plausibly used ---------------------------------
FLAT = {"teams": [
    {"team": {"displayName": "Ohio State"}, "fpi": 28.4},
    {"team": {"displayName": "Michigan"}, "fpi": 16.2},
]}

CATEGORIES = {"teams": [
    {
        "team": {"displayName": "Ohio State", "abbreviation": "OSU"},
        "categories": [
            {"name": "fpi", "values": [{"name": "fpi", "value": 28.4}]},
            {"name": "record", "values": [{"name": "wins", "value": 11}]},
        ],
    },
    {
        "team": {"displayName": "Michigan", "abbreviation": "MICH"},
        "categories": [{"name": "fpi", "values": [{"name": "fpi", "value": 16.2}]}],
    },
]}

STATS_LIST = {"items": [
    {
        "team": {"displayName": "Ohio State"},
        "stats": [
            {"name": "fpiRank", "value": 1},
            {"shortDisplayName": "FPI", "value": 28.4},
        ],
    },
    {
        "team": {"displayName": "Michigan"},
        "stats": [{"shortDisplayName": "FPI", "value": 16.2}],
    },
]}

NESTED_UNDER_KEY = {"powerIndex": {"teams": [
    {"team": {"displayName": "Ohio State"}, "fpi": 28.4},
    {"team": {"displayName": "Michigan"}, "fpi": 16.2},
]}}


class TestParsing:
    @pytest.mark.parametrize(
        "payload", [FLAT, CATEGORIES, STATS_LIST, NESTED_UNDER_KEY],
        ids=["flat", "categories", "stats-list", "nested"],
    )
    def test_all_shapes_yield_the_same_rows(self, payload):
        rows = extract_fpi_rows(payload)
        assert {r["team"] for r in rows} == {"Ohio State", "Michigan"}
        by_team = {r["team"]: r["fpi"] for r in rows}
        assert by_team["Ohio State"] == pytest.approx(28.4)
        assert by_team["Michigan"] == pytest.approx(16.2)

    def test_ranks_are_not_mistaken_for_ratings(self):
        payload = {"teams": [{
            "team": {"displayName": "Ohio State"},
            "categories": [
                {"name": "fpiRank", "values": [{"name": "rank", "value": 1}]},
                {"name": "fpi", "values": [{"name": "fpi", "value": 28.4}]},
            ],
        }]}
        assert extract_fpi_rows(payload)[0]["fpi"] == pytest.approx(28.4)

    def test_implausible_magnitudes_are_rejected(self):
        # A win projection or percentage sitting under an FPI-ish label.
        payload = {"teams": [{"team": {"displayName": "Ohio State"}, "fpi": 9999}]}
        assert extract_fpi_rows(payload) == []

    def test_negative_ratings_are_kept(self):
        payload = {"teams": [{"team": {"displayName": "UMass"}, "fpi": -22.6}]}
        assert extract_fpi_rows(payload)[0]["fpi"] == pytest.approx(-22.6)

    def test_teams_without_a_value_are_skipped(self):
        payload = {"teams": [
            {"team": {"displayName": "Ohio State"}, "fpi": 28.4},
            {"team": {"displayName": "Nowhere"}},
        ]}
        rows = extract_fpi_rows(payload)
        assert [r["team"] for r in rows] == ["Ohio State"]

    def test_duplicates_are_dropped(self):
        payload = {"teams": [
            {"team": {"displayName": "Ohio State"}, "fpi": 28.4},
            {"team": {"displayName": "Ohio State"}, "fpi": 27.0},
        ]}
        assert len(extract_fpi_rows(payload)) == 1

    def test_unrecognisable_payloads_yield_nothing(self):
        for payload in ({}, [], {"nonsense": 1}, None, "a string"):
            assert extract_fpi_rows(payload) == []

    def test_booleans_are_not_treated_as_numbers(self):
        payload = {"teams": [{"team": {"displayName": "Ohio State"}, "fpi": True}]}
        assert extract_fpi_rows(payload) == []


class TestFetch:
    def test_http_error_returns_empty(self):
        class Response:
            status_code = 503

            def json(self):  # pragma: no cover - never reached
                raise AssertionError("should not parse an error response")

        class Session:
            def get(self, *a, **k):
                return Response()

        assert fetch_fpi(SEASON, session=Session()) == []

    def test_network_failure_returns_empty(self):
        import requests

        class Session:
            def get(self, *a, **k):
                raise requests.ConnectionError("blocked")

        assert fetch_fpi(SEASON, session=Session()) == []

    def test_bad_json_returns_empty(self):
        class Response:
            status_code = 200

            def json(self):
                raise ValueError("not json")

        class Session:
            def get(self, *a, **k):
                return Response()

        assert fetch_fpi(SEASON, session=Session()) == []

    def test_season_is_passed_through(self):
        captured = {}

        class Response:
            status_code = 200

            def json(self):
                return FLAT

        class Session:
            def get(self, url, params=None, **k):
                captured.update(params or {})
                return Response()

        fetch_fpi(SEASON, session=Session())
        assert captured["season"] == SEASON


class TestFallbackIntegration:
    @pytest.fixture(autouse=True)
    def quiet(self):
        logging.disable(logging.CRITICAL)
        yield
        logging.disable(logging.NOTSET)

    def _espn_rows(self):
        return [{"team": t[0], "fpi": t[3]} for t in TEAMS]

    def test_fallback_fills_fpi_when_cfbd_has_none(self, monkeypatch):
        monkeypatch.setattr("cfbmeta.sources.espn.fetch_fpi", lambda season, **k: self._espn_rows())
        book = build_rating_book(PreseasonClient(), SEASON, week=0)

        assert "fpi" in book.usable_sources()
        assert book.rating("Ohio State", "fpi") == pytest.approx(26.1)
        assert book.provenance["fpi"].origin == "ESPN"
        assert "via ESPN" in book.provenance["fpi"].describe()

    def test_cfbd_stays_primary_when_it_has_data(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "cfbmeta.sources.espn.fetch_fpi",
            lambda season, **k: called.append(season) or self._espn_rows(),
        )
        book = build_rating_book(FakeClient(), 2025, week=6)
        assert book.provenance["fpi"].origin == "CFBD"
        assert called == [], "ESPN must not be called when CFBD supplied FPI"

    def test_fallback_can_be_disabled(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "cfbmeta.sources.espn.fetch_fpi", lambda season, **k: called.append(season) or []
        )
        build_rating_book(PreseasonClient(), SEASON, week=0, espn_fpi_fallback=False)
        assert called == []

    def test_unknown_espn_teams_do_not_create_phantom_entries(self, monkeypatch):
        rows = self._espn_rows() + [{"team": "Some D2 School", "fpi": 5.0}]
        monkeypatch.setattr("cfbmeta.sources.espn.fetch_fpi", lambda season, **k: rows)
        book = build_rating_book(PreseasonClient(), SEASON, week=0)
        assert normalize_team("Some D2 School") not in book.teams

    def test_a_failing_fallback_leaves_fpi_unusable(self, monkeypatch):
        monkeypatch.setattr(
            "cfbmeta.sources.espn.fetch_fpi",
            lambda season, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        book = build_rating_book(PreseasonClient(), SEASON, week=0)
        assert "fpi" not in book.usable_sources()
        assert book.usable_sources() == ["sp_plus", "talent"]

    def test_no_matches_leaves_fpi_unusable_and_says_so(self, monkeypatch):
        monkeypatch.setattr(
            "cfbmeta.sources.espn.fetch_fpi",
            lambda season, **k: [{"team": "Nobody U", "fpi": 3.0}],
        )
        book = build_rating_book(PreseasonClient(), SEASON, week=0)
        assert "fpi" not in book.usable_sources()
