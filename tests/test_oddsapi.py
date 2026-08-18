"""Offline tests for the second odds feed.

Nothing here touches the network. The payload fixture mirrors the shape
the-odds-api.com returns for americanfootball_ncaaf: an event carries
`home_team`/`away_team` as full names with mascots, and each bookmaker
carries its own `spreads` and `totals` markets.
"""

from __future__ import annotations

import pytest

from cfbmeta.sources import oddsapi
from cfbmeta.sources.market import build_market_map


def event(home, away, books):
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": key,
                "title": key.title(),
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "point": spread, "price": -110},
                            {"name": away, "point": -spread, "price": -110},
                        ],
                    },
                    {"key": "totals", "outcomes": [{"name": "Over", "point": total}]},
                ],
            }
            for key, spread, total in books
        ],
    }


@pytest.fixture
def payload():
    return [
        event("Alabama Crimson Tide", "Georgia Bulldogs", [
            ("pinnacle", -3.0, 51.5),
            ("fanduel", -2.5, 51.0),
            ("draftkings", -3.5, 52.0),
        ]),
    ]


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch):
    """A developer's real key must not turn these into live calls."""
    monkeypatch.delenv("ODDS_API_KEY", raising=False)


# -- parsing -----------------------------------------------------------------
def test_parse_reads_spread_from_the_home_side(payload):
    games = oddsapi.parse_odds(payload)
    assert len(games) == 1
    quotes = {q.book: q.spread for q in games[0].quotes}
    assert quotes == {"pinnacle": -3.0, "fanduel": -2.5, "draftkings": -3.5}
    assert games[0].quotes[0].total == 51.5


def test_parse_skips_events_missing_a_team():
    assert oddsapi.parse_odds([{"home_team": "Alabama", "bookmakers": []}]) == []


def test_parse_skips_books_that_quoted_no_spread():
    games = oddsapi.parse_odds([{
        "home_team": "Alabama", "away_team": "Georgia",
        "bookmakers": [
            {"key": "novig", "markets": [{"key": "totals",
                                          "outcomes": [{"point": 50}]}]},
            {"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [
                {"name": "Alabama", "point": -3.0}]}]},
        ],
    }])
    assert [q.book for q in games[0].quotes] == ["pinnacle"]


def test_sharp_book_wins_over_retail(payload):
    game = oddsapi.parse_odds(payload)[0]
    assert game.sharp_spread().book == "pinnacle"


def test_no_sharp_book_reports_none():
    game = oddsapi.parse_odds([event("Alabama", "Georgia", [("fanduel", -3.0, 50)])])[0]
    assert game.sharp_spread() is None


# -- name matching -----------------------------------------------------------
@pytest.mark.parametrize("odds_name,expected", [
    ("Alabama Crimson Tide", "alabama"),
    ("Ohio State Buckeyes", "ohio state"),
    ("Alabama", "alabama"),
    # normalize_team spells "&" out, so both sides agree on "a and m".
    ("Texas A&M Aggies", "texas a and m"),
    ("Miami (FL) Hurricanes", "miami"),
])
def test_match_team_strips_the_mascot(odds_name, expected):
    known = {"alabama", "ohio state", "ohio", "texas a and m", "miami", "miami oh"}
    assert oddsapi._match_team(odds_name, known) == expected


def test_match_team_prefers_the_longer_school_name():
    # "Ohio" must not swallow "Ohio State Buckeyes".
    assert oddsapi._match_team("Ohio State Buckeyes", {"ohio", "ohio state"}) == "ohio state"


def test_match_team_gives_up_rather_than_guessing():
    assert oddsapi._match_team("Sam Houston Bearkats", {"alabama"}) is None


def test_book_id_collapses_two_feeds_spellings():
    assert oddsapi._book_id("ESPN Bet") == oddsapi._book_id("espnbet")


# -- merging -----------------------------------------------------------------
@pytest.fixture
def markets():
    return build_market_map([{
        "id": 401,
        "homeTeam": "Alabama",
        "awayTeam": "Georgia",
        "lines": [
            {"provider": "DraftKings", "spread": -3.5, "overUnder": 52.0, "spreadOpen": -3.0},
            {"provider": "Bovada", "spread": -3.0, "overUnder": 51.5},
        ],
    }])


def test_merge_benchmarks_against_the_sharp_book(markets, payload):
    assert oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload)) == 1
    assert markets[401]["spread"] == -3.0
    assert markets[401]["sharp_book"] == "pinnacle"
    assert "sharp" in markets[401]["provider"]


def test_merge_can_leave_the_median_benchmark_alone(markets, payload):
    before = markets[401]["spread"]
    oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload),
                               prefer_sharp_benchmark=False)
    assert markets[401]["spread"] == before
    assert "sharp_book" not in markets[401]
    # Line shopping still improves even when the benchmark does not move.
    assert markets[401]["best_home"]["provider"] == "fanduel"


def test_merge_takes_the_better_number_for_each_side(markets, payload):
    oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload))
    # Home bettor wants the most points: -2.5 at FanDuel beats CFBD's -3.0.
    assert markets[401]["best_home"] == {"spread": -2.5, "provider": "fanduel"}
    # Away bettor wants the fewest: -3.5 is the same at both feeds, so the
    # incumbent quote is kept rather than churned for a tie.
    assert markets[401]["best_away"]["spread"] == -3.5


def test_merge_keeps_a_better_incumbent_number(markets):
    odds = oddsapi.parse_odds([event("Alabama Crimson Tide", "Georgia Bulldogs",
                                     [("pinnacle", -4.0, 51.0)])])
    oddsapi.merge_into_markets(markets, odds)
    assert markets[401]["best_home"]["spread"] == -3.0
    assert markets[401]["best_home"]["provider"] == "Bovada"


def test_merge_unions_books_rather_than_summing_them(markets, payload):
    oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload))
    # CFBD: DraftKings, Bovada. Odds api: pinnacle, fanduel, draftkings.
    # DraftKings is in both, so four distinct books, not five.
    assert markets[401]["book_count"] == 4
    assert sum("draftking" in b.lower() for b in markets[401]["books"]) == 1


def test_merge_widens_the_spread_range_across_both_feeds(markets, payload):
    oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload))
    assert markets[401]["spread_range"] == (-3.5, -2.5)


def test_merge_updates_every_key_pointing_at_the_game(markets, payload):
    oddsapi.merge_into_markets(markets, oddsapi.parse_odds(payload))
    assert markets[401] is markets[("alabama", "georgia")]
    assert markets[("alabama", "georgia")]["spread"] == -3.0


def test_merge_flips_a_game_the_feeds_disagree_on_hosting(markets):
    """Neutral-site games get opposite home teams; the spread has to flip."""
    odds = oddsapi.parse_odds([event("Georgia Bulldogs", "Alabama Crimson Tide",
                                     [("pinnacle", 3.0, 51.0)])])
    assert oddsapi.merge_into_markets(markets, odds) == 1
    # Georgia -(+3) from Alabama's side is Alabama -3.
    assert markets[401]["spread"] == -3.0


def test_merge_ignores_games_the_slate_does_not_have(markets):
    odds = oddsapi.parse_odds([event("Boise State Broncos", "Fresno State Bulldogs",
                                     [("pinnacle", -7.0, 55.0)])])
    assert oddsapi.merge_into_markets(markets, odds) == 0
    assert markets[401]["spread"] == -3.25


def test_merge_on_an_empty_market_map_is_a_no_op(payload):
    assert oddsapi.merge_into_markets({}, oddsapi.parse_odds(payload)) == 0


# -- fetch / enrich ----------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.text = text
        self.headers = {"x-requests-remaining": "487"}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.response


def test_fetch_without_a_key_makes_no_request():
    session = FakeSession(FakeResponse())
    assert oddsapi.fetch_odds(api_key="", session=session) == []
    assert session.calls == []


def test_fetch_passes_the_key_and_asks_for_both_markets(payload):
    session = FakeSession(FakeResponse(payload=payload))
    games = oddsapi.fetch_odds(api_key="k", session=session)
    url, params = session.calls[0]
    assert oddsapi.SPORT_KEY in url
    assert params["apiKey"] == "k"
    assert params["markets"] == "spreads,totals"
    assert len(games) == 1


def test_fetch_swallows_an_error_status(caplog):
    session = FakeSession(FakeResponse(status_code=401, text="unauthorized"))
    assert oddsapi.fetch_odds(api_key="bad", session=session) == []


def test_enrich_without_a_key_is_inert(markets):
    before = dict(markets[401])
    assert oddsapi.enrich(markets) == 0
    assert markets[401] == before


def test_enrich_fetches_and_merges(markets, payload, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    session = FakeSession(FakeResponse(payload=payload))
    assert oddsapi.enrich(markets, session=session) == 1
    assert markets[401]["sharp_book"] == "pinnacle"
