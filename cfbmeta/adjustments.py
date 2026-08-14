"""Situational adjustments: rest, travel, and short weeks.

These are the small, well-documented effects that ratings systems ignore
because ratings describe team strength, not the circumstances of a particular
Saturday. Each is capped and each is reported line-by-line in the email so a
number can always be traced back to a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

EARTH_RADIUS_MI = 3958.8


def haversine_miles(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def parse_start(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    for parser in (
        lambda t: datetime.fromisoformat(t),
        lambda t: datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f%z"),
        lambda t: datetime.strptime(t, "%Y-%m-%d"),
    ):
        try:
            dt = parser(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


@dataclass
class TeamGeo:
    team: str
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass
class SituationalModel:
    rest_day_value: float = 0.10
    bye_week_bonus: float = 1.0
    short_week_penalty: float = 0.8
    travel_penalty_per_1k_mi: float = 0.35
    rest_cap: float = 2.0
    travel_cap: float = 1.2
    geo: Dict[str, TeamGeo] = field(default_factory=dict)
    venue_geo: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # team key -> sorted list of kickoff datetimes for the season
    schedule: Dict[str, List[datetime]] = field(default_factory=dict)

    # -- rest ---------------------------------------------------------------
    def days_rest(self, team: str, kickoff: Optional[datetime]) -> Optional[float]:
        if kickoff is None:
            return None
        dates = self.schedule.get(normalize_team(team))
        if not dates:
            return None
        previous = [d for d in dates if d < kickoff]
        if not previous:
            return None
        return (kickoff - max(previous)).total_seconds() / 86400.0

    def rest_adjustment(
        self, home_team: str, away_team: str, kickoff: Optional[datetime]
    ) -> Dict[str, float]:
        home_rest = self.days_rest(home_team, kickoff)
        away_rest = self.days_rest(away_team, kickoff)
        if home_rest is None or away_rest is None:
            return {"home_days": home_rest or 0.0, "away_days": away_rest or 0.0, "total": 0.0}

        points = (home_rest - away_rest) * self.rest_day_value

        # A true bye (13+ days) is worth more than the linear term alone.
        if home_rest >= 13 and away_rest < 13:
            points += self.bye_week_bonus
        elif away_rest >= 13 and home_rest < 13:
            points -= self.bye_week_bonus

        # Short week: a midweek game after a Saturday.
        if home_rest <= 5 and away_rest > 5:
            points -= self.short_week_penalty
        elif away_rest <= 5 and home_rest > 5:
            points += self.short_week_penalty

        return {
            "home_days": round(home_rest, 1),
            "away_days": round(away_rest, 1),
            "total": round(max(-self.rest_cap, min(self.rest_cap, points)), 3),
        }

    # -- travel -------------------------------------------------------------
    def travel_miles(
        self, away_team: str, venue_id=None, home_team: Optional[str] = None
    ) -> Optional[float]:
        away = self.geo.get(normalize_team(away_team))
        if not away or away.lat is None or away.lon is None:
            return None

        dest: Optional[Tuple[float, float]] = None
        if venue_id is not None and str(venue_id) in self.venue_geo:
            dest = self.venue_geo[str(venue_id)]
        elif home_team:
            home = self.geo.get(normalize_team(home_team))
            if home and home.lat is not None and home.lon is not None:
                dest = (home.lat, home.lon)
        if dest is None:
            return None
        return haversine_miles(away.lat, away.lon, dest[0], dest[1])

    def travel_adjustment(
        self, home_team: str, away_team: str, venue_id=None, neutral_site: bool = False
    ) -> Dict[str, float]:
        miles = self.travel_miles(away_team, venue_id=venue_id, home_team=home_team)
        if miles is None:
            return {"miles": 0.0, "total": 0.0}
        if neutral_site:
            # On a neutral field both sides travel; only the difference matters.
            home_miles = self.travel_miles(home_team, venue_id=venue_id) or 0.0
            miles = miles - home_miles
        # Short trips are a non-event; the effect shows up on long hauls.
        excess = max(0.0, abs(miles) - 300.0)
        points = excess / 1000.0 * self.travel_penalty_per_1k_mi
        points = min(self.travel_cap, points)
        if miles < 0:
            points = -points
        return {"miles": round(miles, 0), "total": round(points, 3)}

    def for_game(
        self,
        home_team: str,
        away_team: str,
        kickoff: Optional[datetime],
        venue_id=None,
        neutral_site: bool = False,
    ) -> Dict[str, object]:
        rest = self.rest_adjustment(home_team, away_team, kickoff)
        travel = self.travel_adjustment(home_team, away_team, venue_id, neutral_site)
        return {
            "rest": rest,
            "travel": travel,
            "total": round(rest["total"] + travel["total"], 3),
        }


def build_situational_model(
    client,
    season: int,
    rest_day_value: float = 0.10,
    bye_week_bonus: float = 1.0,
    short_week_penalty: float = 0.8,
    travel_penalty_per_1k_mi: float = 0.35,
    rest_cap: float = 2.0,
    travel_cap: float = 1.2,
    games: Optional[Iterable[dict]] = None,
) -> SituationalModel:
    """Build schedule and geography lookups for the season."""
    import logging

    log = logging.getLogger(__name__)
    model = SituationalModel(
        rest_day_value=rest_day_value,
        bye_week_bonus=bye_week_bonus,
        short_week_penalty=short_week_penalty,
        travel_penalty_per_1k_mi=travel_penalty_per_1k_mi,
        rest_cap=rest_cap,
        travel_cap=travel_cap,
    )

    if games is None:
        try:
            games = client.games(season)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load schedule for situational model: %s", exc)
            games = []
    load_schedule(model, games)

    try:
        attach_geography(model, client.venues(), client.teams())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load venue geography: %s", exc)

    return model


def load_schedule(model: SituationalModel, games: Iterable[dict]) -> int:
    count = 0
    for game in games or []:
        kickoff = parse_start(pick(game, "startDate", "start_date", "startTime", "start_time"))
        if kickoff is None:
            continue
        for key in ("homeTeam", "awayTeam"):
            team = pick(game, key)
            if team:
                model.schedule.setdefault(normalize_team(team), []).append(kickoff)
        count += 1
    for dates in model.schedule.values():
        dates.sort()
    return count


def attach_geography(
    model: SituationalModel, venues: Iterable[dict], teams: Iterable[dict]
) -> int:
    coords_by_id: Dict[str, Tuple[float, float]] = {}
    for venue in venues or []:
        lat = pick_float(venue, "latitude", "location.latitude", "location.x")
        lon = pick_float(venue, "longitude", "location.longitude", "location.y")
        vid = pick(venue, "id")
        if lat is None or lon is None or vid is None:
            continue
        coords_by_id[str(vid)] = (lat, lon)
    model.venue_geo.update(coords_by_id)

    attached = 0
    for team in teams or []:
        name = pick(team, "school", "team")
        if not name:
            continue
        lat = pick_float(team, "location.latitude", "latitude")
        lon = pick_float(team, "location.longitude", "longitude")
        if lat is None or lon is None:
            vid = pick(team, "venueId", "venue_id", "location.venueId", "location.venue_id")
            if vid is not None and str(vid) in coords_by_id:
                lat, lon = coords_by_id[str(vid)]
        if lat is None or lon is None:
            continue
        model.geo[normalize_team(name)] = TeamGeo(team=name, lat=lat, lon=lon)
        attached += 1
    return attached
