"""Preseason behaviour.

In August, CFBD has SP+ and talent for the coming season but SRS and Elo have
no games to work from. The dangerous failure is not a missing source — that
degrades cleanly — it is a source that returns a row for every team with all
the values identical. That predicts a tie in every game, and blending it drags
every projection toward pick'em while looking perfectly healthy.
"""

import logging

import pytest

from cfbmeta.config import Config
from cfbmeta.model import blend, component_margins, effective_weights, project_slate
from cfbmeta.ratings import (
    MIN_SOURCE_DISPERSION,
    MIN_SOURCE_TEAMS,
    RatingBook,
    build_rating_book,
)

from conftest import SEASON, TEAMS, FakeClient

NEXT_SEASON = SEASON + 1


class PreseasonClient(FakeClient):
    """August: SP+/talent are published, SRS/Elo are flat, FPI not out yet."""

    def __init__(self, *, flat_srs=True, flat_elo=True, no_fpi=True, **kwargs):
        super().__init__(**kwargs)
        self.flat_srs, self.flat_elo, self.no_fpi = flat_srs, flat_elo, no_fpi

    def srs_ratings(self, year):
        if not self.flat_srs:
            return super().srs_ratings(year)
        return [{"year": year, "team": t[0], "conference": t[1], "rating": 0.0} for t in TEAMS]

    def elo_ratings(self, year, week=None):
        if not self.flat_elo:
            return super().elo_ratings(year, week)
        return [{"year": year, "week": 0, "team": t[0], "elo": 1500} for t in TEAMS]

    def fpi_ratings(self, year):
        return [] if self.no_fpi else super().fpi_ratings(year)


@pytest.fixture(autouse=True)
def quiet_logs():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def preseason_book():
    return build_rating_book(PreseasonClient(), NEXT_SEASON, week=0)


class TestDegenerateSourcePruning:
    def test_flat_sources_are_dropped(self, preseason_book):
        assert preseason_book.rating("Ohio State", "srs") is None
        assert preseason_book.rating("Ohio State", "elo") is None

    def test_real_sources_survive(self, preseason_book):
        assert preseason_book.rating("Ohio State", "sp_plus") == pytest.approx(28.4)
        assert preseason_book.rating("Ohio State", "talent") is not None

    def test_usable_sources_reflects_reality(self, preseason_book):
        assert preseason_book.usable_sources() == ["sp_plus", "talent"]

    def test_flat_source_does_not_drag_the_blend(self, preseason_book):
        """The whole point: a no-opinion source must not pull toward a tie."""
        config = Config()
        weights = effective_weights(config, 0)
        comps = component_margins(preseason_book, "Ohio State", "UMass", weights)

        assert all(c.source not in ("srs", "elo") for c in comps)
        # Every surviving component should be a real, non-zero opinion.
        assert all(abs(c.margin) > 0.001 for c in comps)
        # And the blend should sit near the strong sources, not diluted to zero.
        assert blend(comps) > 45.0

    def test_a_flat_source_would_have_dragged_it(self):
        """Guard against someone removing the pruning: prove it mattered."""
        book = build_rating_book(PreseasonClient(), NEXT_SEASON, week=0)
        config = Config()
        weights = effective_weights(config, 0)
        pruned = blend(component_margins(book, "Ohio State", "UMass", weights))

        # Re-add the flat sources by hand to simulate the unpruned behaviour.
        for entry in book.teams.values():
            entry.values["srs"] = 0.0
            entry.values["elo"] = 0.0
        unpruned = blend(component_margins(book, "Ohio State", "UMass", weights))
        assert pruned > unpruned, "pruning must change the answer, or it is dead code"

    def test_provenance_explains_each_source(self, preseason_book):
        prov = preseason_book.provenance
        assert prov["srs"].usable is False
        assert "no games played yet" in prov["srs"].reason
        assert prov["fpi"].usable is False
        assert "no data" in prov["fpi"].reason
        assert prov["sp_plus"].usable is True
        assert prov["sp_plus"].teams == len(TEAMS)

    def test_provenance_records_the_season_requested(self, preseason_book):
        assert all(p.season == NEXT_SEASON for p in preseason_book.provenance.values())

    def test_provenance_lines_are_human_readable(self, preseason_book):
        text = "\n".join(preseason_book.provenance_lines())
        assert "SP+" in text and str(NEXT_SEASON) in text
        assert "unusable" in text

    def test_sparse_source_is_dropped(self):
        book = RatingBook(NEXT_SEASON)
        book.load_sp([{"team": "Ohio State", "rating": 28.4}])
        book.finalize()
        # A single rated team is not a usable source.
        assert "sp_plus" not in book.usable_sources()
        assert f"only 1 teams" in book.provenance["sp_plus"].reason

    def test_threshold_boundaries_are_sane(self):
        assert MIN_SOURCE_TEAMS >= 2
        assert 0 < MIN_SOURCE_DISPERSION < 5


class TestFullyDegradedSeason:
    def test_no_usable_source_is_reported_not_silently_projected(self):
        class NothingClient(PreseasonClient):
            def sp_ratings(self, year):
                return []

            def talent(self, year):
                return []

        book = build_rating_book(NothingClient(), NEXT_SEASON, week=0)
        assert book.usable_sources() == []

        config = Config()
        games = [g for g in FakeClient().games(SEASON) if g["week"] == 6]
        # Nothing is rated, so nothing should be projected as if it were.
        assert project_slate(games, book, config) == []


class TestPreseasonStillProduces:
    def test_a_full_slate_projects_from_sp_and_talent_alone(self):
        client = PreseasonClient()
        book = build_rating_book(client, NEXT_SEASON, week=0)
        config = Config()
        games = [g for g in client.games(SEASON) if g["week"] == 6]
        projections = project_slate(games, book, config)

        assert len(projections) == len(games)
        for proj in projections:
            assert proj.components, "every game should still have a projection"
            assert {c.source for c in proj.components} <= {"sp_plus", "talent"}

    def test_report_footer_states_which_data_was_used(self):
        from cfbmeta.report import build_context, render_html, render_text

        client = PreseasonClient()
        book = build_rating_book(client, NEXT_SEASON, week=0)
        config = Config()
        games = [g for g in client.games(SEASON) if g["week"] == 6]
        projections = project_slate(games, book, config)

        context = build_context(projections, config, week=0, season=NEXT_SEASON, book=book)
        assert context["provenance"]

        text = render_text(context)
        assert f"DATA (season {NEXT_SEASON})" in text
        assert "unusable" in text

        html = render_html(context)
        assert str(NEXT_SEASON) in html
        assert "{{" not in html

    def test_context_without_a_book_still_renders(self):
        from cfbmeta.report import build_context, render_html

        context = build_context([], Config(), week=0, season=NEXT_SEASON)
        assert context["provenance"] == []
        assert render_html(context)
