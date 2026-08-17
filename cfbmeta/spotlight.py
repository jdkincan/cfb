"""Expanded detail for one followed team, pinned to the top of the readout.

The weekly slate is a scan: fifty rows, sorted by edge. That is the right shape
for finding bets and the wrong shape for the one game you actually care about.
The spotlight is the opposite — everything known about a single matchup, in the
order a person reads it.

Three things it adds over a normal row:

* **Roster** — the top rated players on each side, from the same roster/recruiting
  join that builds the talent score. Names, positions, class year, star rating.
* **Momentum** — how each team's rating has moved week to week, drawn from the
  snapshot archive. This is empty in September and gets more useful every week,
  because it can only be built from history we keep ourselves.
* **The full ledger** — every component and adjustment, not the summary.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

STAR_LABEL = {5: "5★", 4: "4★", 3: "3★", 2: "2★", 1: "1★"}
CLASS_LABEL = {1: "Fr", 2: "So", 3: "Jr", 4: "Sr", 5: "5th"}


@dataclass
class Player:
    name: str
    position: str = ""
    class_year: str = ""
    rating: Optional[float] = None
    stars: Optional[int] = None

    @property
    def stars_label(self) -> str:
        return STAR_LABEL.get(self.stars or 0, "—")


@dataclass
class TeamDetail:
    team: str
    players: List[Player] = field(default_factory=list)
    talent: Optional[float] = None
    blue_chip_ratio: Optional[float] = None
    coach: str = ""
    coach_score: Optional[float] = None
    ratings: Dict[str, float] = field(default_factory=dict)
    momentum: List[tuple] = field(default_factory=list)  # (week, sp_plus)

    @property
    def momentum_delta(self) -> Optional[float]:
        """Change in SP+ from the first archived week to the latest."""
        if len(self.momentum) < 2:
            return None
        return round(self.momentum[-1][1] - self.momentum[0][1], 2)

    @property
    def momentum_recent(self) -> Optional[float]:
        """Change over the last archived week only."""
        if len(self.momentum) < 2:
            return None
        return round(self.momentum[-1][1] - self.momentum[-2][1], 2)


@dataclass
class Spotlight:
    projection: Any
    home: TeamDetail
    away: TeamDetail
    note: str = ""


def top_players(
    client,
    team: str,
    season: int,
    limit: int = 5,
    recruit_lookback: int = 8,
) -> List[Player]:
    """Best-recruited players on a team's current roster.

    Recruiting rating is a proxy for talent, not for current production — a
    redshirt five-star who has never played outranks a productive three-star
    senior here. It answers "who is on this roster" rather than "who is good",
    and should be read that way.
    """
    try:
        roster = client.get("roster", year=season, team=team)
    except Exception as exc:  # noqa: BLE001
        log.warning("no roster for %s: %s", team, exc)
        return []

    ratings: Dict[str, tuple] = {}
    for year in range(season - recruit_lookback + 1, season + 1):
        try:
            rows = client.get("recruiting_players", year=year)
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            athlete = pick(row, "athleteId", "athlete_id")
            rating = pick_float(row, "rating")
            if athlete and rating is not None:
                ratings[str(athlete)] = (rating, pick_float(row, "stars"))

    players: List[Player] = []
    for entry in roster:
        athlete = str(pick(entry, "id", "athleteId") or "")
        found = ratings.get(athlete)
        if not found:
            continue
        rating, stars = found
        first = pick(entry, "firstName", "first_name", default="") or ""
        last = pick(entry, "lastName", "last_name", default="") or ""
        year_value = pick_float(entry, "year")
        players.append(Player(
            name=f"{first} {last}".strip() or pick(entry, "name", default="") or "unknown",
            position=pick(entry, "position", default="") or "",
            class_year=CLASS_LABEL.get(int(year_value) if year_value else 0, ""),
            rating=rating,
            stars=int(stars) if stars is not None else None,
        ))

    players.sort(key=lambda p: p.rating or 0.0, reverse=True)
    return players[:limit]


def rating_momentum(
    team: str, season: int, source: str = "sp_plus", archive_root=None
) -> List[tuple]:
    """(week, rating) for a team across every archived week this season.

    Reads the snapshot archive, so it is empty until runs have accumulated.
    That is the point: CFBD cannot answer this question at all.
    """
    from .archive import list_snapshots

    key = normalize_team(team)
    series: List[tuple] = []
    for directory in list_snapshots(season, archive_root):
        path = directory / "ratings.csv"
        if not path.exists():
            continue
        try:
            week = int(directory.name.split("-")[-1])
        except ValueError:
            continue
        try:
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    if row.get("team_key") == key and row.get("source") == source:
                        series.append((week, round(float(row["rating"]), 2)))
                        break
        except (OSError, ValueError, KeyError):
            continue
    series.sort()
    return series


def build_spotlight(
    projections: List[Any],
    team: str,
    client=None,
    season: Optional[int] = None,
    book=None,
    coach_model=None,
    player_limit: int = 5,
    archive_root=None,
    all_games: Optional[List[dict]] = None,
    project_one=None,
) -> Optional[Spotlight]:
    """Find the followed team's game this week and gather everything about it.

    Falls back to the unfiltered schedule when the team is missing from the
    slate. The bettable slate excludes FCS opponents, but "show me my team
    every week" means every week — including the September cupcake, which is
    exactly the game the slate drops.
    """
    if not team:
        return None
    key = normalize_team(team)
    match = next(
        (p for p in projections
         if normalize_team(p.home_team) == key or normalize_team(p.away_team) == key),
        None,
    )

    note = ""
    if match is None and all_games and project_one is not None:
        raw = next(
            (g for g in all_games
             if normalize_team(pick(g, "homeTeam", "home_team") or "") == key
             or normalize_team(pick(g, "awayTeam", "away_team") or "") == key),
            None,
        )
        if raw is not None:
            try:
                match = project_one(raw)
                note = (
                    "Not on the bettable slate — non-FBS opponent, so there is no "
                    "market to price against."
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not project the spotlight game: %s", exc)
    if match is None:
        return None

    season = season or match.season

    def detail(name: str) -> TeamDetail:
        entry = book.get(name) if book is not None else None
        info = TeamDetail(
            team=name,
            talent=(entry.values.get("talent") if entry else None),
            blue_chip_ratio=(entry.meta.get("blue_chip_ratio") if entry else None),
            ratings=dict(entry.values) if entry else {},
        )
        if coach_model is not None:
            info.coach = coach_model.coach_for(name) or ""
            info.coach_score = round(coach_model.team_score(name), 2)
        if client is not None:
            info.players = top_players(client, name, season, player_limit)
        info.momentum = rating_momentum(name, season, archive_root=archive_root)
        return info

    spot = Spotlight(
        projection=match,
        home=detail(match.home_team),
        away=detail(match.away_team),
        note=note,
    )
    if not spot.home.momentum and not spot.away.momentum:
        extra = (
            "Rating momentum needs archived weeks to compare against; it fills in "
            "as the season runs."
        )
        spot.note = f"{spot.note} {extra}".strip()
    return spot
