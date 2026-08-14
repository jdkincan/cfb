"""Offline fixtures.

The sandbox this was built in cannot reach the CFBD API, and more importantly
CI shouldn't depend on a third-party API being up. Everything below is
synthetic but shaped like the real payloads.

``FakeClient`` deliberately serves *camelCase* for some endpoints and
*snake_case* for others: CFBD has shipped both across API versions, and the
tests should prove the client tolerates either.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any, Dict, List

import pytest

from cfbmeta.config import Config

SEASON = 2025

TEAMS = [
    # (school, conference, sp+, fpi, elo, srs, talent, venue_id)
    ("Ohio State", "Big Ten", 28.4, 26.1, 2180, 21.5, 985.2, 1),
    ("Georgia", "SEC", 26.9, 25.0, 2150, 20.8, 978.4, 2),
    ("Texas", "SEC", 24.1, 23.4, 2110, 19.2, 962.1, 3),
    ("Alabama", "SEC", 23.5, 22.8, 2095, 18.9, 970.3, 4),
    ("Penn State", "Big Ten", 21.2, 20.5, 2060, 17.1, 921.7, 5),
    ("Oregon", "Big Ten", 20.8, 19.9, 2050, 16.8, 918.0, 6),
    ("Notre Dame", "FBS Independents", 18.4, 18.0, 2020, 15.2, 905.5, 7),
    ("Michigan", "Big Ten", 16.2, 15.4, 1990, 13.4, 930.2, 8),
    ("Tennessee", "SEC", 14.9, 14.2, 1965, 12.1, 899.8, 9),
    ("Kansas State", "Big 12", 11.3, 10.8, 1920, 9.4, 812.4, 10),
    ("Utah", "Big 12", 9.8, 9.1, 1900, 8.2, 805.1, 11),
    ("Iowa State", "Big 12", 8.1, 7.6, 1880, 6.9, 790.3, 12),
    ("Colorado", "Big 12", 4.2, 3.8, 1820, 3.1, 830.6, 13),
    ("Wyoming", "Mountain West", -6.5, -7.1, 1650, -5.8, 620.4, 14),
    ("Air Force", "Mountain West", -3.1, -3.8, 1700, -2.5, 640.2, 15),
    ("Florida", "SEC", 6.4, 6.0, 1860, 5.2, 915.4, 16),
    ("Hawai'i", "Mountain West", -12.4, -13.0, 1560, -11.2, 580.1, 17),
    ("UMass", "MAC", -22.6, -23.4, 1420, -20.1, 505.3, 18),
]

VENUES = [
    # (id, name, city, lat, lon, elevation_ft)
    (1, "Ohio Stadium", "Columbus", 40.0017, -83.0197, 725),
    (2, "Sanford Stadium", "Athens", 33.9497, -83.3733, 630),
    (3, "DKR Texas Memorial", "Austin", 30.2837, -97.7325, 500),
    (4, "Bryant-Denny Stadium", "Tuscaloosa", 33.2083, -87.5504, 220),
    (5, "Beaver Stadium", "University Park", 40.8122, -77.8560, 1150),
    (6, "Autzen Stadium", "Eugene", 44.0583, -123.0681, 400),
    (7, "Notre Dame Stadium", "Notre Dame", 41.6983, -86.2338, 720),
    (8, "Michigan Stadium", "Ann Arbor", 42.2658, -83.7487, 860),
    (9, "Neyland Stadium", "Knoxville", 35.9550, -83.9250, 830),
    (10, "Bill Snyder Family Stadium", "Manhattan", 39.2020, -96.5940, 1060),
    (11, "Rice-Eccles Stadium", "Salt Lake City", 40.7600, -111.8487, 4637),
    (12, "Jack Trice Stadium", "Ames", 42.0140, -93.6360, 940),
    (13, "Folsom Field", "Boulder", 40.0094, -105.2669, 5360),
    (14, "War Memorial Stadium", "Laramie", 41.3114, -105.5666, 7220),
    (15, "Falcon Stadium", "Colorado Springs", 38.9970, -104.8434, 6621),
    (16, "Ben Hill Griffin Stadium", "Gainesville", 29.6500, -82.3486, 65),
    (17, "Ching Complex", "Honolulu", 21.2969, -157.8175, 60),
    (18, "McGuirk Stadium", "Amherst", 42.3897, -72.5311, 260),
]

COACHES = {
    "Ohio State": "Ryan Day",
    "Georgia": "Kirby Smart",
    "Texas": "Steve Sarkisian",
    "Alabama": "Kalen DeBoer",
    "Penn State": "James Franklin",
    "Oregon": "Dan Lanning",
    "Notre Dame": "Marcus Freeman",
    "Michigan": "Sherrone Moore",
    "Tennessee": "Josh Heupel",
    "Kansas State": "Chris Klieman",
    "Utah": "Kyle Whittingham",
    "Iowa State": "Matt Campbell",
    "Colorado": "Deion Sanders",
    "Wyoming": "Jay Sawvel",
    "Air Force": "Troy Calhoun",
    "Florida": "Billy Napier",
    "Hawai'i": "Timmy Chang",
    "UMass": "Joe Harasymiak",
}

# Schools that changed coaches ahead of the current season, and who held the
# job before. Needed so first-year-hire detection has something real to find.
PRIOR_COACHES = {"UMass": "Don Brown"}


def _team_row(name: str) -> tuple:
    for row in TEAMS:
        if row[0] == name:
            return row
    raise KeyError(name)


class FakeClient:
    """Stand-in for CFBDClient with deterministic synthetic payloads."""

    def __init__(self, season: int = SEASON, fail: set[str] | None = None) -> None:
        self.season = season
        self.fail = fail or set()
        self.calls: List[str] = []
        self._rng = random.Random(1234)

    def _guard(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"simulated failure in {name}")

    # -- ratings: camelCase, like CFBD v2 ------------------------------------
    def sp_ratings(self, year: int) -> List[Dict[str, Any]]:
        self._guard("sp")
        return [
            {
                "year": year,
                "team": t[0],
                "conference": t[1],
                "rating": t[2],
                # SP+ quotes defense as points allowed: lower is better.
                "offense": {"rating": 30.0 + t[2] / 2.0, "ranking": 1},
                "defense": {"rating": 25.0 - t[2] / 2.0, "ranking": 1},
                "specialTeams": {"rating": 0.1},
            }
            for t in TEAMS
        ]

    def fpi_ratings(self, year: int) -> List[Dict[str, Any]]:
        self._guard("fpi")
        return [
            {
                "year": year,
                "team": t[0],
                "conference": t[1],
                "fpi": t[3],
                "efficiencies": {"overall": t[3], "offense": 60.0, "defense": 55.0},
            }
            for t in TEAMS
        ]

    # -- ratings: snake_case, like CFBD v1 -----------------------------------
    def srs_ratings(self, year: int) -> List[Dict[str, Any]]:
        self._guard("srs")
        return [{"year": year, "team": t[0], "conference": t[1], "rating": t[5]} for t in TEAMS]

    def elo_ratings(self, year: int, week=None) -> List[Dict[str, Any]]:
        self._guard("elo")
        rows = []
        for t in TEAMS:
            # Two weeks of history; the later one should win.
            rows.append({"year": year, "week": 1, "team": t[0], "elo": t[4] - 25})
            rows.append({"year": year, "week": 5, "team": t[0], "elo": t[4]})
        return rows

    def talent(self, year: int) -> List[Dict[str, Any]]:
        self._guard("talent")
        return [{"year": year, "school": t[0], "talent": t[6]} for t in TEAMS]

    def returning_production(self, year: int) -> List[Dict[str, Any]]:
        self._guard("returning")
        return [{"season": year, "team": t[0], "total_ppa": 0.55} for t in TEAMS]

    def teams(self, year=None) -> List[Dict[str, Any]]:
        self._guard("teams")
        out = []
        for t in TEAMS:
            venue = next(v for v in VENUES if v[0] == t[7])
            out.append(
                {
                    "id": t[7],
                    "school": t[0],
                    "conference": t[1],
                    "venue_id": t[7],
                    "location": {
                        "venue_id": t[7],
                        "name": venue[1],
                        "latitude": venue[3],
                        "longitude": venue[4],
                        "elevation": venue[5],
                    },
                }
            )
        return out

    def venues(self) -> List[Dict[str, Any]]:
        self._guard("venues")
        return [
            {
                "id": v[0],
                "name": v[1],
                "city": v[2],
                "latitude": v[3],
                "longitude": v[4],
                "elevation": v[5],
            }
            for v in VENUES
        ]

    def coaches(self, year=None) -> List[Dict[str, Any]]:
        """One row per coach holding a job in ``year``.

        Schools in PRIOR_COACHES show their predecessor in earlier seasons, so
        first-year-hire detection has a genuine handover to find.
        """
        self._guard("coaches")
        rows = []
        for school, coach in COACHES.items():
            if school in PRIOR_COACHES and year < self.season:
                coach = PRIOR_COACHES[school]
            t = _team_row(school)
            first, _, last = coach.partition(" ")
            rows.append(
                {
                    "firstName": first,
                    "lastName": last,
                    "seasons": [
                        {
                            "school": school,
                            "year": year,
                            "games": 12,
                            "wins": 8,
                            "losses": 4,
                            "spOverall": t[2],
                            "spOffense": 30.0,
                            "spDefense": 25.0,
                        }
                    ],
                }
            )
        return rows

    def games(self, year: int, week=None, season_type: str = "regular") -> List[Dict[str, Any]]:
        self._guard("games")
        return [g for g in build_games(year) if week is None or g["week"] == week]

    def lines(self, year: int, week=None, season_type: str = "regular") -> List[Dict[str, Any]]:
        self._guard("lines")
        return build_lines(year, week)

    def calendar(self, year: int) -> List[Dict[str, Any]]:
        self._guard("calendar")
        out = []
        start = dt.datetime(year, 8, 24, tzinfo=dt.timezone.utc)
        for week in range(1, 15):
            first = start + dt.timedelta(days=7 * (week - 1))
            out.append(
                {
                    "season": year,
                    "week": week,
                    "seasonType": "regular",
                    "firstGameStart": first.isoformat(),
                    "lastGameStart": (first + dt.timedelta(days=6)).isoformat(),
                }
            )
        return out


WEEK6_MATCHUPS = [
    # (home, away, neutral)
    ("Ohio State", "Michigan", False),
    ("Georgia", "Alabama", False),
    ("Texas", "Tennessee", False),
    ("Wyoming", "Florida", False),  # altitude + long travel
    ("Colorado", "Air Force", False),  # altitude, but visitor is also high
    ("Penn State", "Iowa State", False),
    ("Oregon", "Utah", False),
    ("Notre Dame", "Kansas State", True),  # neutral site
    ("Hawai'i", "UMass", False),  # extreme travel
]


def build_games(year: int) -> List[Dict[str, Any]]:
    """A few weeks of schedule so rest-day logic has history to work with."""
    games: List[Dict[str, Any]] = []
    gid = 4000
    base = dt.datetime(year, 8, 30, 19, 0, tzinfo=dt.timezone.utc)

    # Weeks 1-5: everyone plays a filler game each week.
    for week in range(1, 6):
        kickoff = base + dt.timedelta(days=7 * (week - 1))
        for i in range(0, len(TEAMS) - 1, 2):
            home, away = TEAMS[i][0], TEAMS[i + 1][0]
            gid += 1
            games.append(
                {
                    "id": gid,
                    "season": year,
                    "week": week,
                    "seasonType": "regular",
                    "startDate": kickoff.isoformat(),
                    "homeTeam": home,
                    "awayTeam": away,
                    "homePoints": 28,
                    "awayPoints": 21,
                    "neutralSite": False,
                    "conferenceGame": True,
                    "venueId": _team_row(home)[7],
                    "venue": next(v[1] for v in VENUES if v[0] == _team_row(home)[7]),
                }
            )

    # Week 6: the slate under test, no scores yet.
    kickoff = base + dt.timedelta(days=35)
    for home, away, neutral in WEEK6_MATCHUPS:
        gid += 1
        games.append(
            {
                "id": gid,
                "season": year,
                "week": 6,
                "seasonType": "regular",
                "startDate": kickoff.isoformat(),
                "homeTeam": home,
                "awayTeam": away,
                "homePoints": None,
                "awayPoints": None,
                "neutralSite": neutral,
                "conferenceGame": False,
                "venueId": _team_row(home)[7],
                "venue": next(v[1] for v in VENUES if v[0] == _team_row(home)[7]),
            }
        )
    return games


def build_lines(year: int, week=None) -> List[Dict[str, Any]]:
    """Market lines close to, but not identical to, the model's view."""
    offsets = {
        "Ohio State": -1.5,
        "Georgia": 0.0,
        "Texas": 4.0,  # market well off our number -> should surface as a play
        "Wyoming": -3.5,
        "Colorado": 0.5,
        "Penn State": 0.0,
        "Oregon": -6.0,  # big disagreement the other way
        "Notre Dame": 1.0,
        "Hawai'i": 0.0,
    }
    out = []
    for game in build_games(year):
        if week is not None and game["week"] != week:
            continue
        if game["week"] != 6:
            continue
        home, away = game["homeTeam"], game["awayTeam"]
        raw = _team_row(home)[2] - _team_row(away)[2] + 2.4
        spread = -(raw + offsets.get(home, 0.0))
        out.append(
            {
                "id": game["id"],
                "season": year,
                "week": game["week"],
                "homeTeam": home,
                "awayTeam": away,
                "lines": [
                    {"provider": "DraftKings", "spread": round(spread * 2) / 2,
                     "overUnder": 52.5, "homeMoneyline": -200, "awayMoneyline": 170},
                    {"provider": "Bovada", "spread": round(spread * 2) / 2 + 0.5,
                     "overUnder": 53.0},
                    {"provider": "consensus", "spread": round(spread * 2) / 2,
                     "overUnder": 52.5},
                ],
            }
        )
    return out


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.season = SEASON
    cfg.cache_dir = ""
    return cfg


@pytest.fixture
def book(client):
    from cfbmeta.ratings import build_rating_book

    return build_rating_book(client, SEASON, week=6)


@pytest.fixture
def week6_games(client):
    return [g for g in client.games(SEASON) if g["week"] == 6]
