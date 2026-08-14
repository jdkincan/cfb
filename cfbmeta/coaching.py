"""Coaching value, measured as performance against roster talent.

"Good coach" is easy to assert and hard to quantify. The measurable version:
fit how much SP+ rating a team's recruiting talent buys on average, then credit
a coach with the residual — how much better or worse their teams finish than
their raw material says they should. Do it across a career, weight recent
seasons more, and shrink hard toward zero so a coach with one good year doesn't
get treated as a three-point edge.

Two components come out of this:

* ``career`` - talent-adjusted residual in SP+ points, exponentially weighted
  toward recent seasons and shrunk by career length.
* ``first_year`` - a penalty for a coach in year one at a school, where scheme
  turnover and roster fit reliably cost a team points.

The whole thing is capped (``coach_adj_cap``, default 1.5 points). It is a
tiebreaker on close games, not a driver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

# How fast older seasons stop mattering. 0.75 => a season 4 years back carries
# ~32% the weight of last season.
RECENCY_DECAY = 0.75
# Pseudo-seasons of "average coach" mixed in. 3 is heavy shrinkage on purpose.
CAREER_SHRINK_SEASONS = 3.0
FIRST_YEAR_PENALTY = 1.2


@dataclass
class CoachSeason:
    coach: str
    school: str
    year: int
    sp_overall: Optional[float] = None
    talent_z: Optional[float] = None
    wins: int = 0
    losses: int = 0

    @property
    def key(self) -> str:
        return normalize_coach(self.coach)


@dataclass
class CoachProfile:
    name: str
    seasons: List[CoachSeason] = field(default_factory=list)
    residual: float = 0.0  # talent-adjusted, weighted, shrunk
    seasons_counted: int = 0

    @property
    def schools(self) -> List[str]:
        return sorted({s.school for s in self.seasons})


def normalize_coach(name: str) -> str:
    return " ".join(str(name or "").lower().split())


@dataclass
class CoachModel:
    profiles: Dict[str, CoachProfile] = field(default_factory=dict)
    team_coach: Dict[str, str] = field(default_factory=dict)  # team key -> coach name
    first_year_teams: Dict[str, bool] = field(default_factory=dict)
    cap: float = 1.5
    first_year_penalty: float = FIRST_YEAR_PENALTY
    # Fitted talent -> SP+ relationship, kept for reporting/debugging.
    talent_slope: float = 0.0
    talent_intercept: float = 0.0

    def coach_for(self, team: str) -> Optional[str]:
        return self.team_coach.get(normalize_team(team))

    def team_score(self, team: str) -> float:
        """Coaching value for a team in points, capped and including year one."""
        coach = self.coach_for(team)
        score = 0.0
        if coach:
            prof = self.profiles.get(normalize_coach(coach))
            if prof:
                score += prof.residual
        if self.first_year_teams.get(normalize_team(team)):
            score -= self.first_year_penalty
        return max(-self.cap, min(self.cap, score))

    def for_game(self, home_team: str, away_team: str) -> Dict[str, float]:
        home = self.team_score(home_team)
        away = self.team_score(away_team)
        net = max(-self.cap, min(self.cap, home - away))
        return {
            "home": round(home, 3),
            "away": round(away, 3),
            "total": round(net, 3),
            "home_coach": self.coach_for(home_team) or "unknown",
            "away_coach": self.coach_for(away_team) or "unknown",
        }


def _fit_linear(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Least-squares slope/intercept for y ~ x. Returns (slope, intercept)."""
    n = len(points)
    if n < 10:
        return 0.0, 0.0
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    if sxx <= 0:
        return 0.0, my
    slope = sxy / sxx
    return slope, my - slope * mx


def _zscore_map(raw: Dict[str, float]) -> Dict[str, float]:
    if len(raw) < 2:
        return {k: 0.0 for k in raw}
    vals = list(raw.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    sd = var ** 0.5
    if sd <= 0:
        return {k: 0.0 for k in raw}
    return {k: (v - mean) / sd for k, v in raw.items()}


def build_coach_model(
    client,
    current_season: int,
    lookback_years: int = 6,
    cap: float = 1.5,
) -> CoachModel:
    """Assemble coach profiles from CFBD coach records, talent and SP+."""
    model = CoachModel(cap=cap)
    seasons = list(range(current_season - lookback_years, current_season + 1))

    # Talent and SP+ per season, so a coach-season can be scored in context.
    talent_by_year: Dict[int, Dict[str, float]] = {}
    sp_by_year: Dict[int, Dict[str, float]] = {}
    for year in seasons:
        try:
            raw = {
                normalize_team(pick(r, "team", "school")): pick_float(r, "talent")
                for r in client.talent(year)
                if pick(r, "team", "school") and pick_float(r, "talent") is not None
            }
            talent_by_year[year] = _zscore_map(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("talent unavailable for %d: %s", year, exc)
        try:
            sp_by_year[year] = {
                normalize_team(pick(r, "team", "school")): pick_float(r, "rating")
                for r in client.sp_ratings(year)
                if pick(r, "team", "school") and pick_float(r, "rating") is not None
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("SP+ unavailable for %d: %s", year, exc)

    coach_seasons: List[CoachSeason] = []
    for year in seasons:
        try:
            rows = client.coaches(year)
        except Exception as exc:  # noqa: BLE001
            log.warning("coaches unavailable for %d: %s", year, exc)
            continue
        coach_seasons.extend(_parse_coach_rows(rows, year, talent_by_year, sp_by_year))

    if not coach_seasons:
        log.warning("no coach data available; coaching adjustment will be zero")
        return model

    # Fit SP+ ~ talent_z across every observed team-season.
    fit_points = [
        (cs.talent_z, cs.sp_overall)
        for cs in coach_seasons
        if cs.talent_z is not None and cs.sp_overall is not None
    ]
    slope, intercept = _fit_linear(fit_points)
    model.talent_slope, model.talent_intercept = slope, intercept
    log.info("talent->SP+ fit: sp = %.2f * talent_z + %.2f (n=%d)", slope, intercept, len(fit_points))

    for cs in coach_seasons:
        model.profiles.setdefault(cs.key, CoachProfile(name=cs.coach)).seasons.append(cs)

    for prof in model.profiles.values():
        prof.residual, prof.seasons_counted = _career_residual(
            prof.seasons, current_season, slope, intercept
        )

    _assign_current_teams(model, coach_seasons, current_season)
    return model


def _parse_coach_rows(
    rows: Iterable[dict],
    year: int,
    talent_by_year: Dict[int, Dict[str, float]],
    sp_by_year: Dict[int, Dict[str, float]],
) -> List[CoachSeason]:
    out: List[CoachSeason] = []
    for row in rows or []:
        first = pick(row, "firstName", "first_name", default="") or ""
        last = pick(row, "lastName", "last_name", default="") or ""
        name = pick(row, "name", default="") or f"{first} {last}".strip()
        if not name:
            continue
        for season in pick(row, "seasons", default=[]) or []:
            syear = pick(season, "year")
            school = pick(season, "school", "team")
            if not school or syear is None or int(syear) != int(year):
                continue
            tkey = normalize_team(school)
            sp = pick_float(season, "spOverall", "sp_overall")
            if sp is None:
                sp = sp_by_year.get(year, {}).get(tkey)
            out.append(
                CoachSeason(
                    coach=name,
                    school=school,
                    year=int(syear),
                    sp_overall=sp,
                    talent_z=talent_by_year.get(year, {}).get(tkey),
                    wins=int(pick_float(season, "wins", default=0) or 0),
                    losses=int(pick_float(season, "losses", default=0) or 0),
                )
            )
    return out


def _career_residual(
    seasons: List[CoachSeason], current_season: int, slope: float, intercept: float
) -> Tuple[float, int]:
    """Recency-weighted, shrunk average of talent-adjusted SP+ residuals."""
    num = den = 0.0
    counted = 0
    for cs in seasons:
        if cs.sp_overall is None or cs.talent_z is None:
            continue
        if cs.year >= current_season:
            # The current season is incomplete; it can't grade a coach yet.
            continue
        expected = slope * cs.talent_z + intercept
        residual = cs.sp_overall - expected
        weight = RECENCY_DECAY ** max(0, current_season - 1 - cs.year)
        num += residual * weight
        den += weight
        counted += 1

    if den <= 0:
        return 0.0, 0
    weighted = num / den
    # Shrink toward zero (an average coach) by effective career length.
    shrunk = weighted * den / (den + CAREER_SHRINK_SEASONS)
    return round(shrunk, 3), counted


def _assign_current_teams(
    model: CoachModel, coach_seasons: List[CoachSeason], current_season: int
) -> None:
    """Map each team to its current coach and flag first-year hires."""
    current = [cs for cs in coach_seasons if cs.year == current_season]
    prior = {
        normalize_team(cs.school): cs.key
        for cs in coach_seasons
        if cs.year == current_season - 1
    }
    for cs in current:
        tkey = normalize_team(cs.school)
        model.team_coach[tkey] = cs.coach
        model.first_year_teams[tkey] = prior.get(tkey) != cs.key
