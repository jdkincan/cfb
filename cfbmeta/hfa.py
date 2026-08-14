"""Per-venue home field advantage.

Every rating system already bakes an average HFA into its ratings, but home
field is emphatically not uniform: Autzen and a 7,200-foot afternoon in Laramie
are not worth what a half-empty Saturday in a dome is worth.

The estimator is the mean *residual against a rating*: for every non-neutral
game, how much did the home team beat the margin its rating implied?

    residual = actual_margin - (rating_home - rating_away)
    raw_hfa(t) = mean residual across t's home games

Controlling for opponent strength is not optional here. An earlier version of
this used each team as its own control — mean home margin minus mean away
margin, halved — on the theory that team quality cancels out. It does not.
College teams do not face equal opposition home and away: they host the weaker
non-conference games and travel for the harder ones. Measured across four
seasons of FBS play that estimator returned **4.4 points**, against a true
value near 2.6, and would have added nearly two phantom points to every home
team on the slate.

The residual method returns 2.70 on the same games, and conference-only games —
where schedules are naturally balanced — independently agree at 2.58.

Because per-team residuals are still noisy, the raw figure is shrunk hard
toward the league mean with a pseudo-count (``hfa_shrink_games``, default 60
games), so a team with two flukey home wins doesn't get a six-point home field.

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


# Below this many home games, a venue residual is noise; use the league mean.
MIN_HOME_GAMES = 6


@dataclass
class VenueProfile:
    team: str
    home_games: int = 0
    residual_sum: float = 0.0
    elevation_ft: Optional[float] = None

    @property
    def raw_hfa(self) -> Optional[float]:
        """Mean points beaten by, above what the ratings implied, at home."""
        if self.home_games < MIN_HOME_GAMES:
            return None
        return self.residual_sum / self.home_games

    @property
    def sample_size(self) -> int:
        return self.home_games


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

    used = 0
    for season in seasons:
        try:
            games = client.games(season)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load %d games for HFA: %s", season, exc)
            continue
        try:
            ratings = {
                normalize_team(pick(r, "team", "school")): pick_float(r, "rating")
                for r in client.sp_ratings(season)
                if pick(r, "team", "school") and pick_float(r, "rating") is not None
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("no %d ratings for HFA control: %s", season, exc)
            continue
        used += ingest_games(model, games, ratings)
    log.info("HFA estimated from %d rating-controlled home games", used)

    try:
        attach_elevations(model, client.venues(), client.teams())
    except Exception as exc:  # noqa: BLE001
        log.warning("could not attach venue elevations: %s", exc)

    # Re-centre the league mean on the pooled residual across every game,
    # which is a far better estimate than averaging per-team means (those give
    # a team with 6 home games the same say as one with 30).
    total_games = sum(p.home_games for p in model.profiles.values())
    if total_games >= 400:
        pooled = sum(p.residual_sum for p in model.profiles.values()) / total_games
        model.league_hfa = round(pooled, 3)
        log.info(
            "league HFA measured at %.2f from %d games across %d venues",
            model.league_hfa, total_games, len(model.profiles),
        )
    return model


def ingest_games(
    model: HFAModel,
    games: Iterable[dict],
    ratings: Optional[Dict[str, float]] = None,
) -> int:
    """Accumulate rating-controlled home residuals from completed games.

    ``ratings`` maps a normalized team name to a points-scale rating for the
    season those games belong to. A game whose teams aren't both rated is
    skipped: without a rating there is no way to separate home field from
    having hosted a weaker opponent, and counting it anyway is exactly the bias
    this estimator exists to avoid.
    """
    ratings = ratings or {}
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

        hkey, akey = normalize_team(home), normalize_team(away)
        home_rating, away_rating = ratings.get(hkey), ratings.get(akey)
        if home_rating is None or away_rating is None:
            continue

        residual = (hp - ap) - (home_rating - away_rating)
        prof = model.profiles.setdefault(hkey, VenueProfile(team=home))
        prof.home_games += 1
        prof.residual_sum += residual
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
