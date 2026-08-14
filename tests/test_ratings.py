import pytest

from cfbmeta.ratings import RatingBook, build_rating_book, normalize_team
from cfbmeta.sources.cfbd import pick, pick_float

from conftest import SEASON, FakeClient


class TestNormalizeTeam:
    def test_state_abbreviations_collapse(self):
        assert normalize_team("Ohio St.") == normalize_team("Ohio State")
        assert normalize_team("Penn St") == normalize_team("Penn State")

    def test_case_and_punctuation_insensitive(self):
        assert normalize_team("Hawai'i") == normalize_team("Hawaii")
        assert normalize_team("TEXAS A&M") == normalize_team("Texas A and M")

    def test_accents_are_stripped(self):
        assert normalize_team("San José State") == normalize_team("San Jose State")

    def test_empty_is_safe(self):
        assert normalize_team("") == ""
        assert normalize_team(None) == ""


class TestFieldAccess:
    """The client must tolerate both API casings."""

    def test_camel_and_snake_resolve_identically(self):
        camel = {"homeTeam": "Texas", "startDate": "2025-10-04"}
        snake = {"home_team": "Texas", "start_date": "2025-10-04"}
        for row in (camel, snake):
            assert pick(row, "homeTeam") == "Texas"
            assert pick(row, "home_team") == "Texas"
            assert pick(row, "startDate") == "2025-10-04"

    def test_nested_dotted_lookup(self):
        row = {"offense": {"rating": 31.4}}
        assert pick_float(row, "offense.rating") == pytest.approx(31.4)

    def test_missing_returns_default(self):
        assert pick({}, "nope", default="fallback") == "fallback"
        assert pick_float({}, "nope", default=1.5) == pytest.approx(1.5)

    def test_none_values_fall_through_to_default(self):
        assert pick({"spread": None}, "spread", default=7) == 7
        assert pick_float({"rating": ""}, "rating", default=0.0) == 0.0

    def test_non_numeric_is_not_coerced(self):
        assert pick_float({"rating": "abc"}, "rating", default=None) is None


class TestRatingBook:
    def test_all_sources_load(self, book):
        assert len(book) == 18
        osu = book.get("Ohio State")
        assert osu is not None
        for source in ("sp_plus", "fpi", "elo", "srs", "talent"):
            assert osu.get(source) is not None, f"{source} missing"

    def test_native_point_sources_pass_through_unchanged(self, book):
        assert book.rating("Ohio State", "sp_plus") == pytest.approx(28.4)
        assert book.rating("Ohio State", "fpi") == pytest.approx(26.1)
        assert book.rating("Ohio State", "srs") == pytest.approx(21.5)

    def test_scaled_sources_land_on_the_points_scale(self, book):
        # Elo is z-scored to the SP+ spread, so the best team should end up in
        # the same neighbourhood as its SP+ rating rather than near 2000.
        elo = book.rating("Ohio State", "elo")
        assert 10 < elo < 45
        assert book.rating("UMass", "elo") < -10

    def test_elo_uses_the_latest_week(self, book):
        # Fixture has week 1 at elo-25 and week 5 at elo; week 5 must win.
        assert book.get("Ohio State").meta["elo_raw"] == pytest.approx(2180)

    def test_scaling_is_independent_of_load_order(self, client):
        forward = RatingBook(SEASON)
        forward.load_sp(client.sp_ratings(SEASON))
        forward.load_elo(client.elo_ratings(SEASON))
        forward.finalize()

        backward = RatingBook(SEASON)
        backward.load_elo(client.elo_ratings(SEASON))
        backward.load_sp(client.sp_ratings(SEASON))
        backward.finalize()

        assert forward.rating("Georgia", "elo") == pytest.approx(
            backward.rating("Georgia", "elo")
        )

    def test_sp_offense_defense_stored_as_meta(self, book):
        meta = book.get("Ohio State").meta
        assert "sp_offense" in meta and "sp_defense" in meta

    def test_ordering_is_preserved_within_a_source(self, book):
        assert book.rating("Ohio State", "sp_plus") > book.rating("Michigan", "sp_plus")
        assert book.rating("Ohio State", "elo") > book.rating("Michigan", "elo")

    def test_unknown_team_returns_none(self, book):
        assert book.get("Not A Real School") is None
        assert book.rating("Not A Real School", "sp_plus") is None

    def test_conference_is_captured(self, book):
        assert book.get("Georgia").conference == "SEC"


class TestDegradedSources:
    def test_a_failing_source_does_not_kill_the_book(self):
        client = FakeClient(fail={"fpi"})
        book = build_rating_book(client, SEASON, week=6)
        assert len(book) == 18
        assert book.rating("Ohio State", "fpi") is None
        assert book.rating("Ohio State", "sp_plus") is not None

    def test_missing_sp_falls_back_to_default_scale(self):
        client = FakeClient(fail={"sp"})
        book = build_rating_book(client, SEASON, week=6)
        # Elo still needs to be scaled to something sane.
        assert book.rating("Ohio State", "elo") is not None
        assert abs(book.rating("Ohio State", "elo")) < 60
