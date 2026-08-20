"""Tests for scraping posted win totals across books."""

from __future__ import annotations

import pytest

from cfbmeta.sources.wintotal_scrape import (
    ALIASES,
    BookTotal,
    ScrapedTotal,
    _match,
    book_names,
    estimate_under_price,
    load,
    parse,
    probability_to_american,
    to_lines,
)

HEADER = ("<thead><tr><th>Time</th><th>BetMGM</th><th>DraftKings</th>"
          "<th>Caesars</th></tr></thead>")


def row(name, *pairs):
    cells = "".join(
        f'<td><span class="data-value"> o{t} </span>'
        f'<span class="data-value"> {p} </span></td>' for t, p in pairs)
    return f'<tr class="" data-name="{name}"><td>{name}</td>{cells}</tr>'


# -- price maths -------------------------------------------------------------
@pytest.mark.parametrize("price", [-250, -150, -110, 100, 145, 300])
def test_american_round_trips_through_probability(price):
    from cfbmeta.probability import implied_probability
    assert probability_to_american(implied_probability(price)) == pytest.approx(price, abs=1)


def test_estimated_under_sits_opposite_the_over():
    """A juiced over implies a cheap under."""
    assert estimate_under_price(-200) > 0
    assert estimate_under_price(200) < 0


def test_the_assumed_hold_shows_up_in_the_estimate():
    from cfbmeta.probability import implied_probability
    over = -120
    under = estimate_under_price(over, hold=0.045)
    total = implied_probability(over) + implied_probability(under)
    assert total == pytest.approx(1.045, abs=0.01)


# -- parsing -----------------------------------------------------------------
def test_parse_reads_one_row_per_team():
    html = HEADER + row("georgia", ("10.5", "-125"), ("9.5", "+110"))
    scraped = parse(html)
    assert len(scraped) == 1
    assert scraped[0].team == "georgia"
    assert [q.total for q in scraped[0].quotes] == [10.5, 9.5]


def test_parse_attributes_quotes_to_books():
    html = HEADER + row("georgia", ("10.5", "-125"), ("9.5", "+110"))
    assert [q.book for q in parse(html)[0].quotes] == ["BetMGM", "DraftKings"]


def test_parse_skips_malformed_cells():
    html = HEADER + ('<tr data-name="x"><td><span class="data-value"> - </span>'
                     '<span class="data-value"> - </span></td></tr>')
    assert parse(html) == []


def test_parse_marks_every_under_price_as_estimated():
    html = HEADER + row("georgia", ("10.5", "-125"))
    assert all(q.under_estimated for q in parse(html)[0].quotes)


def test_book_names_drops_the_non_book_columns():
    assert book_names(HEADER) == ["BetMGM", "DraftKings", "Caesars"]


def test_book_names_on_a_page_without_a_header():
    assert book_names("<table></table>") == []


# -- line shopping -----------------------------------------------------------
def test_the_over_takes_the_lowest_number_on_the_board():
    entry = ScrapedTotal("georgia", [
        BookTotal("A", 10.5, -110), BookTotal("B", 9.5, -110)])
    assert entry.best_over().total == 9.5


def test_the_under_takes_the_highest_number_on_the_board():
    entry = ScrapedTotal("georgia", [
        BookTotal("A", 10.5, -110, -105), BookTotal("B", 9.5, -110, -105)])
    assert entry.best_under().total == 10.5


def test_price_breaks_a_tie_on_the_number():
    entry = ScrapedTotal("georgia", [
        BookTotal("A", 9.5, -130), BookTotal("B", 9.5, +100)])
    assert entry.best_over().over_price == 100


def test_disagreement_is_reported():
    same = ScrapedTotal("x", [BookTotal("A", 9.5, -110), BookTotal("B", 9.5, -110)])
    differ = ScrapedTotal("y", [BookTotal("A", 9.5, -110), BookTotal("B", 10.5, -110)])
    assert not same.totals_disagree
    assert differ.totals_disagree


# -- name matching -----------------------------------------------------------
def test_a_school_name_that_prefixes_another_is_not_swallowed():
    """Regression: ULM's total was being handed to the Ragin' Cajuns.

    "louisiana-monroe" starts with "louisiana ", so a bare prefix rule matched
    the wrong school and invented a three-win edge out of nothing.
    """
    known = {"louisiana", "ul monroe", "louisiana tech"}
    assert _match("louisiana-monroe", known) == "ul monroe"
    assert _match("louisiana", known) == "louisiana"


@pytest.mark.parametrize("source,expected", [
    ("appalachian state", "app state"),
    ("umass", "massachusetts"),
    ("miami (fl)", "miami"),
])
def test_known_spelling_differences_are_aliased(source, expected):
    assert _match(source, {expected}) == expected


def test_an_alias_pointing_at_an_unrated_team_is_not_forced():
    assert _match("umass", {"georgia"}) is None


def test_mascots_are_still_stripped():
    assert _match("Alabama Crimson Tide", {"alabama"}) == "alabama"


def test_an_unknown_team_matches_nothing():
    assert _match("Somewhere Tech", {"alabama"}) is None


def test_every_alias_target_is_lowercase_and_normalized():
    from cfbmeta.ratings import normalize_team
    for source, target in ALIASES.items():
        assert normalize_team(source) == source
        assert normalize_team(target) == target


# -- conversion --------------------------------------------------------------
def test_two_rows_matching_one_team_are_both_dropped():
    """A collision means a bad match, and a bad match invents edge."""
    scraped = [ScrapedTotal("louisiana", [BookTotal("A", 7.5, -110, -105)]),
               ScrapedTotal("louisiana ragin cajuns", [BookTotal("A", 3.5, -110, -105)])]
    assert to_lines(scraped, {"louisiana"}) == []


def test_each_side_keeps_its_own_number_and_book():
    scraped = [ScrapedTotal("georgia", [
        BookTotal("BetMGM", 9.5, -130, -102, under_estimated=True),
        BookTotal("Caesars", 10.5, +105, -120, under_estimated=True)])]
    line = to_lines(scraped, {"georgia"})[0]
    assert (line.total, line.book) == (9.5, "BetMGM")
    assert (line.under_number, line.under_book) == (10.5, "Caesars")
    assert line.under_price_estimated


def test_unmatched_teams_are_dropped_not_guessed():
    scraped = [ScrapedTotal("nowhere state", [BookTotal("A", 5.5, -110, -105)])]
    assert to_lines(scraped, {"georgia"}) == []


def test_a_row_with_no_quotes_is_skipped():
    assert to_lines([ScrapedTotal("georgia", [])], {"georgia"}) == []


# -- failure handling --------------------------------------------------------
def test_load_is_never_fatal():
    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("network is down")
    assert load(known_teams={"georgia"}, session=Boom()) == []


def test_a_non_200_response_is_reported_not_parsed():
    class NotFound:
        def get(self, *a, **k):
            class R:
                status_code = 404
                text = ""
            return R()
    assert load(known_teams={"georgia"}, session=NotFound()) == []
