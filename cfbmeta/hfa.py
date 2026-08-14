"""Per-venue home field advantage.

Every rating system already bakes an average HFA into its ratings, but home
field is emphatically not uniform: Autzen and a 7,200-foot afternoon in Laramie
are not worth what a half-empty Saturday in a dome is worth.

The estimator uses each team as its own control. For team ``t``:

    raw_hfa(t) = (mean margin in home games - mean margin in away games) / 2

Team quality cancels out of the difference, so what's left is mostly venue
effect plus noise. Because it *is* noisy, the raw figure is shrunk hard toward
the league mean with a pseudo-count (``hfa_shrink_games``, default 60 games —
roughly ten seasons), so a team with two flukey home blowouts doesn't get
credited with a six-point home field.

Elevation is handled separately, as a differential against where the visitor
plays its own home games: Wyoming hosting Air Force is not the altitude edge
that Wyoming hosting Florida is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)


@dataclass
class VenueProfile:
    team: str
    home_games: int = 0
    home_margin_sum: float = 0.0
    away_games: int = 0
    away_margin_sum: float = 0.0
    elevation_ft: Optional[float] = None

    @property
    def home_margin_avg(self) -> float:
        return self.home_margin_sum / self.home_games if self.home_games else 0.0

    @property
    def away_margin_avg(self) -> float:
        return self.away_margin_sum / self.away_games if self.away_games else 0.0

    @property
    def raw_hfa(self) -> Optional[float]:
        if self.home_games < 3 or self.away_games < 3:
            return None
        return (self.home_margin_avg - self.away_margin_avg) / 2.0

    @property
    def sample_size(self) -> int:
        return min(self.home_games, self.away_games)


@dataclass
class HFAModel:
    league_hfa: float = 2.35
    shrink_games: float = 60.0
    hfa_min: float = 0.0
    hfa_max: float = 5.0
    altitude_bonus_per_1k_ft: float = 0.22
    altitude_threshold_ft: float = 3000.0
    profiles: Dict[str, VenueProfile] = field(default_factory=dict)

    def profile(self, team: str) -> Optional[VenueProfile]:
        return self.profiles.get(normalize_team(team))

    def team_hfa(self, team: str) -> float:
        """Shrunk, clamped home field advantage in points for a team's venue."""
        prof = self.profile(team)
        if not prof or prof.raw_hfa is None:
            return self.league_hfa
        n = float(prof.sample_size)
        shrunk = (n * prof.raw_hfa + self.shrink_games * self.league_hfa) / (
            n + self.shrink_games
        )
        return max(self.hfa_min, min(self.hfa_max, shrunk))

    def elevation_edge(self, home_team: str, away_team: str) -> float:
        """Extra points for dragging a sea-level team up a mountain."""
        home = self.profile(home_team)
        away = self.profile(away_team)
        if not home or home.elevation_ft is None:
            return 0.0
        if home.elevation_ft < self.altitude_threshold_ft:
            return 0.0
        away_elev = away.elevation_ft if away and away.elevation_ft is not None else 500.0
        delta_ft = home.elevation_ft - away_elev
        if delta_ft <= 0:
            return 0.0
        return round(delta_ft / 1000.0 * self.altitude_bonus_per_1k_ft, 3)

    def for_game(self, home_team: str, away_team: str, neutral_site: bool) -> Dict[str, float]:
        """Full home-field breakdown for one game, in points on the home side."""
        if neutral_site:
            return {"base": 0.0, "elevation": 0.0, "total": 0.0}
        base = self.team_hfa(home_team)
        elev = self.elevation_edge(home_team, away_team)
        return {
            "base": round(base, 3),
            "elevation": elev,
            "total": round(min(self.hfa_max + 1.5, base + elev), 3),
        }


def build_hfa_model(
    client,
    seasons: Iterable[int],
    league_hfa: float = 2.35,
    shrink_games: float = 60.0,
    hfa_min: float = 0.0,
    hfa_max: float = 5.0,
    altitude_bonus_per_1k_ft: float = 0.22,
    altitude_threshold_ft: float = 3000.0,
) -> HFAModel:
    """Estimate venue HFA from completed games across several seasons."""
    model = HFAModel(
        league_hfa=league_hfa,
        shrink_games=shrink_games,
        hfa_min=hfa_min,
        hfa_max=hfa_max,
        altitude_bonus_per_1k_ft=altitude_bonus_per_1k_ft,
        altitude_threshold_ft=altitude_threshold_ft,
    )

    all_games: List[dict] = []
    for season in seasons:
        try:
            all_games.extend(client.games(season))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load %d games for HFA: %s", season, exc)

    ingest_games(model, all_games)

    try:
        attach_elevations(model, client.venues(), client.teams())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not attach venue elevations: %s", exc)

    measured = [p.raw_hfa for p in model.profiles.values() if p.raw_hfa is not None]
    if len(measured) >= 40:
        # Re-center the league mean on what the data actually shows, so the
        # shrinkage target isn't a stale constant.
        model.league_hfa = round(sum(measured) / len(measured), 3)
        log.info("league HFA measured at %.2f from %d teams", model.league_hfa, len(measured))
    return model


def ingest_games(model: HFAModel, games: Iterable[dict]) -> int:
    """Accumulate home/away margins per team from completed, non-neutral games."""
    used = 0
    for game in games:
        if pick(game, "neutralSite", "neutral_site", default=False):
            continue
        home = pick(game, "homeTeam", "home_team", "home")
        away = pick(game, "awayTeam", "away_team", "away")
        hp = pick_float(game, "homePoints", "home_points", "homeScore", "home_score")
        ap = pick_float(game, "awayPoints", "away_points", "awayScore", "away_score")
        if not home or not away or hp is None or ap is None:
            continue

        margin = hp - ap
        hkey, akey = normalize_team(home), normalize_team(away)
        hprof = model.profiles.setdefault(hkey, VenueProfile(team=home))
        aprof = model.profiles.setdefault(akey, VenueProfile(team=away))
        hprof.home_games += 1
        hprof.home_margin_sum += margin
        aprof.away_games += 1
        aprof.away_margin_sum += -margin
        used += 1
    return used


def attach_elevations(
    model: HFAModel, venues: Iterable[dict], teams: Iterable[dict]
) -> int:
    """Map each team to the elevation of its home venue."""
    by_id: Dict[str, float] = {}
    by_name: Dict[str, float] = {}
    for venue in venues or []:
        elev = pick_float(venue, "elevation")
        if elev is None:
            continue
        vid = pick(venue, "id")
        if vid is not None:
            by_id[str(vid)] = elev
        name = pick(venue, "name")
        if name:
            by_name[normalize_team(name)] = elev

    attached = 0
    for team in teams or []:
        name = pick(team, "school", "team")
        if not name:
            continue
        prof = model.profiles.setdefault(normalize_team(name), VenueProfile(team=name))
        vid = pick(team, "venueId", "venue_id")
        elev = by_id.get(str(vid)) if vid is not None else None
        if elev is None:
            venue_name = pick(team, "venue", "location.venue", "location.name")
            if venue_name:
                elev = by_name.get(normalize_team(venue_name))
        if elev is None:
            elev = pick_float(team, "location.elevation", "elevation")
        if elev is not None:
            prof.elevation_ft = elev
            attached += 1
    return attached
