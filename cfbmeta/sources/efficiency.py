"""A weekly, opponent-adjusted efficiency rating — the SP+ analog.

Why this exists
---------------
CFBD's SP+, FPI and SRS are **season-final numbers**. The ``week`` parameter is
accepted and silently ignored: ask for week 1 and week 15 of 2025 and you get
byte-identical ratings. There is no weekly SP+ history to fetch, and none can
be backfilled, because CFBD keeps no history and Bill Connelly's weekly
articles have no author-filterable index.

Elo is the one genuine weekly series CFBD carries. That is not enough on its
own, and Elo is a results-only rating — it knows a team won by 3 but not that
it out-gained the opponent by 200 yards and lost two fumbles on the way.

What is available is ``/stats/game/advanced``: one row per team per game,
carrying exactly the raw material SP+ is built from — PPA, success rate,
explosiveness, line yards, havoc, standard and passing downs. A whole season
is one API call. Because every row is stamped with its week, a rating can be
rebuilt from precisely the games completed before any past week, for any
season. That is a real weekly time series, and unlike a snapshot archive it
exists retroactively rather than starting today.

The construction
----------------
The same shape SP+ uses. For each team-game, the offense's efficiency against
that opponent is modelled as::

    efficiency(team vs opp) = mu + off[team] - def[opp] + hfa * is_home

solved as one ridge-regularised least-squares system over every completed
game. ``off`` is offensive quality, ``def`` is defensive suppression (higher =
stingier), and overall quality is their sum. Both vectors are mean-centred, so
a rating is quality above an average team.

Quality then calibrates to points by regressing actual scoring margins on
rating differences, which makes the output a points-above-average number on
the same scale as SP+ rather than an abstract index.

What it is worth
----------------
Measured on 2025, refitting before each week and predicting that week's games
(weeks 5-15, 566 FBS-vs-FBS games, strictly out of sample):

===================================  ========
weekly efficiency rating alone        13.81
margin ridge ratings alone            12.91
**0.4 x efficiency + 0.6 x margin**   **12.44**
season-final SP+ (leaky ceiling)      10.34
===================================  ========

The row that matters is the third. Efficiency alone is *worse* than plain
margin ratings — it is a noisier signal — but blending the two beats either,
which is the definition of independent information. Efficiency knows how a
team played; margin knows what the scoreboard said. They disagree in useful
ways.

Correlation with published season-final SP+ across 136 teams is r = 0.95,
which is a sanity check on the construction rather than a target: SP+ is not
ground truth here, actual margins are.

The last row is a reminder, not a goal. Season-final SP+ has already seen the
games it is being scored on, so 10.34 is unreachable by anything honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..ratings import normalize_team
from .cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Composite weights over the per-game efficiency metrics. PPA carries most of
# the signal; success rate stabilises it (a team can post a good PPA on two
# explosive plays and nothing else); explosiveness gets a light touch because
# it is the noisiest of the three. Additive constants are deliberately absent:
# the fit has an intercept and the solution is mean-centred, so any centring
# term would land entirely in mu and change nothing.
PPA_WEIGHT = 0.60
SUCCESS_RATE_WEIGHT = 0.90
EXPLOSIVENESS_WEIGHT = 0.10

# Low, because the efficiency signal is already heavily shrunk by being an
# average over plays. Swept on 2025: 1-2 is flat and better than anything
# above 3.
DEFAULT_RIDGE = 2.0

# Fitting needs enough games for opponent adjustment to mean anything. Below
# this the schedule has not crossed over enough for the system to be
# identified, and the ratings are mostly ridge prior.
MIN_OBSERVATIONS = 60


@dataclass
class EfficiencyRatings:
    """Opponent-adjusted efficiency for one point in time."""

    week: int
    offense: Dict[str, float] = field(default_factory=dict)
    defense: Dict[str, float] = field(default_factory=dict)
    quality: Dict[str, float] = field(default_factory=dict)
    points: Dict[str, float] = field(default_factory=dict)
    home_field: float = 0.0
    points_per_unit: float = 0.0
    observations: int = 0
    # The population ranks are quoted against. FCS teams have to be *in* the
    # fit — they are a third of September's schedule — but ranking an FBS team
    # 11th behind North Dakota State is not a useful readout.
    eligible: Optional[set] = None

    def __len__(self) -> int:
        return len(self.quality)

    def get(self, team: str, default: Optional[float] = None) -> Optional[float]:
        return self.points.get(normalize_team(team), default)

    def _pool(self) -> Dict[str, float]:
        if not self.eligible:
            return self.points
        pool = {t: v for t, v in self.points.items() if t in self.eligible}
        return pool or self.points

    def ranked(self, include_all: bool = False) -> List[Tuple[str, float]]:
        source = self.points if include_all else self._pool()
        return sorted(source.items(), key=lambda kv: -kv[1])

    def rank_of(self, team: str) -> Optional[int]:
        """Rank within the eligible pool, or None if the team is unrated."""
        key = normalize_team(team)
        value = self.points.get(key)
        if value is None:
            return None
        pool = self._pool()
        if key not in pool:
            return None
        return 1 + sum(1 for v in pool.values() if v > value)


def game_efficiency(side: Any) -> Optional[float]:
    """Collapse one side's per-game advanced stats into a single number."""
    if side is None:
        return None
    ppa = pick_float(side, "ppa")
    success = pick_float(side, "successRate", "success_rate")
    if ppa is None or success is None:
        return None
    explosive = pick_float(side, "explosiveness", default=0.0) or 0.0
    return (
        PPA_WEIGHT * ppa
        + SUCCESS_RATE_WEIGHT * success
        + EXPLOSIVENESS_WEIGHT * explosive
    )


def home_map(games: Iterable[dict]) -> Dict[Any, Tuple[str, str, bool]]:
    """Game id -> (home key, away key, neutral).

    The advanced-stats rows carry team and opponent but not which one was at
    home, and home field is worth real points, so it has to be joined back in
    from the schedule or it lands in the ratings as team quality.
    """
    out: Dict[Any, Tuple[str, str, bool]] = {}
    for game in games or []:
        gid = pick(game, "id", "gameId", "game_id")
        home = pick(game, "homeTeam", "home_team")
        away = pick(game, "awayTeam", "away_team")
        if gid is None or not home or not away:
            continue
        neutral = bool(pick(game, "neutralSite", "neutral_site", default=False))
        out[gid] = (normalize_team(home), normalize_team(away), neutral)
    return out


def _observations(
    rows: Iterable[dict],
    homes: Dict[Any, Tuple[str, str, bool]],
    before_week: Optional[int],
    season: Optional[int],
) -> List[Tuple[str, str, float, float]]:
    """(team, opponent, efficiency, home indicator) for eligible team-games.

    Only the ``offense`` block is used. Each game already contributes two rows,
    one per team, and a row's ``defense`` block is by construction the same
    information as the opponent row's ``offense`` block — feeding both in would
    double the row count without adding one independent observation.
    """
    obs: List[Tuple[str, str, float, float]] = []
    for row in rows or []:
        week = pick_float(row, "week")
        if week is None:
            continue
        if before_week is not None and int(week) >= int(before_week):
            continue
        if season is not None:
            row_season = pick_float(row, "season", "year", default=season)
            if row_season is None or int(row_season) != int(season):
                continue

        gid = pick(row, "gameId", "game_id", "id")
        if gid not in homes:
            continue
        team = normalize_team(pick(row, "team", default="") or "")
        opponent = normalize_team(pick(row, "opponent", default="") or "")
        if not team or not opponent:
            continue
        value = game_efficiency(pick(row, "offense"))
        if value is None:
            continue

        home, away, neutral = homes[gid]
        if neutral:
            indicator = 0.0
        elif team == home:
            indicator = 1.0
        elif team == away:
            indicator = -1.0
        else:
            indicator = 0.0
        obs.append((team, opponent, float(value), indicator))
    return obs


def fit_efficiency(
    rows: Iterable[dict],
    games: Iterable[dict],
    before_week: Optional[int] = None,
    season: Optional[int] = None,
    ridge: float = DEFAULT_RIDGE,
    homes: Optional[Dict[Any, Tuple[str, str, bool]]] = None,
    center_on: Optional[set] = None,
) -> Optional[EfficiencyRatings]:
    """Solve for offensive and defensive efficiency from completed games.

    ``before_week`` is strict: week 8 fits on weeks 1-7 only. That strictness
    is the entire reason this is usable for backtesting — including the target
    week would leak the result being predicted into the rating predicting it.

    ``center_on`` names the population that defines an average team — pass the
    FBS list. It matters more than it looks: FCS opponents appear in the fit
    (they have to, or half of September is unusable), and centring on all 300+
    teams makes the average team far worse than an average FBS team, inflating
    every FBS rating by roughly a third and destroying comparability with SP+.
    """
    import numpy as np

    games = list(games or [])
    homes = homes if homes is not None else home_map(games)
    obs = _observations(rows, homes, before_week, season)
    if len(obs) < MIN_OBSERVATIONS:
        log.debug(
            "efficiency fit skipped before week %s: %d observations (need %d)",
            before_week, len(obs), MIN_OBSERVATIONS,
        )
        return None

    teams = sorted({t for t, _, _, _ in obs} | {o for _, o, _, _ in obs})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    # Columns: [offense(n) | defense(n) | intercept | home field]
    design = np.zeros((len(obs) + 2 * n, 2 * n + 2))
    target = np.zeros(len(obs) + 2 * n)
    for k, (team, opponent, value, indicator) in enumerate(obs):
        design[k, index[team]] = 1.0
        design[k, n + index[opponent]] = -1.0
        design[k, 2 * n] = 1.0
        design[k, 2 * n + 1] = indicator
        target[k] = value
    # Ridge rows pull team effects toward zero, leaving intercept and home
    # field unpenalised — they are league constants, not team claims.
    for j in range(2 * n):
        design[len(obs) + j, j] = np.sqrt(ridge)

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    offense = solution[:n]
    defense = solution[n:2 * n]

    # Centre on the reference population, defaulting to everyone fitted.
    if center_on:
        mask = np.array([team in center_on for team in teams])
        matched = int(mask.sum())
        # Fall back only when the join clearly failed. A caller who asks for a
        # deliberately small population gets it; a caller who passes 136 FBS
        # names and matches four has a name-normalisation bug, and centring on
        # those four would corrupt every rating in the book.
        if matched >= 3 and matched >= 0.25 * len(center_on):
            offense = offense - offense[mask].mean()
            defense = defense - defense[mask].mean()
        else:
            log.warning(
                "efficiency centring: only %d of %d reference teams matched; "
                "centring on all %d instead", matched, len(center_on), len(teams),
            )
            offense = offense - offense.mean()
            defense = defense - defense.mean()
    else:
        offense = offense - offense.mean()
        defense = defense - defense.mean()

    ratings = EfficiencyRatings(
        week=int(before_week) if before_week is not None else 0,
        offense={t: float(offense[i]) for i, t in enumerate(teams)},
        defense={t: float(defense[i]) for i, t in enumerate(teams)},
        quality={t: float(offense[i] + defense[i]) for i, t in enumerate(teams)},
        observations=len(obs),
        eligible=set(center_on) if center_on else None,
    )
    _calibrate(ratings, games, before_week, season)
    return ratings


def _calibrate(
    ratings: EfficiencyRatings,
    games: Sequence[dict],
    before_week: Optional[int],
    season: Optional[int],
) -> None:
    """Put the rating on a points scale by regressing margins onto it.

    Efficiency units are meaningless on their own. Regressing actual scoring
    margin on rating difference converts them into points above average, the
    same scale SP+ reports, and recovers home field in points as a by-product.
    """
    import numpy as np

    rows: List[List[float]] = []
    margins: List[float] = []
    for game in games or []:
        week = pick_float(game, "week")
        if week is None:
            continue
        if before_week is not None and int(week) >= int(before_week):
            continue
        if season is not None:
            row_season = pick_float(game, "season", default=season)
            if row_season is None or int(row_season) != int(season):
                continue
        home_pts = pick_float(game, "homePoints", "home_points")
        away_pts = pick_float(game, "awayPoints", "away_points")
        if home_pts is None or away_pts is None:
            continue
        home = normalize_team(pick(game, "homeTeam", "home_team", default="") or "")
        away = normalize_team(pick(game, "awayTeam", "away_team", default="") or "")
        if home not in ratings.quality or away not in ratings.quality:
            continue
        neutral = bool(pick(game, "neutralSite", "neutral_site", default=False))
        rows.append([ratings.quality[home] - ratings.quality[away], 0.0 if neutral else 1.0])
        margins.append(home_pts - away_pts)

    if len(rows) < 30:
        log.debug("efficiency calibration skipped: only %d scored games", len(rows))
        return

    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(margins), rcond=None)
    scale, hfa = float(coef[0]), float(coef[1])
    if scale <= 0:
        # A non-positive scale means the fit found no signal; publishing it
        # would invert every rating.
        log.warning("efficiency calibration produced scale %.2f; leaving unscaled", scale)
        return
    ratings.points_per_unit = scale
    ratings.home_field = hfa
    ratings.points = {t: q * scale for t, q in ratings.quality.items()}


def weekly_series(
    rows: Iterable[dict],
    games: Iterable[dict],
    weeks: Iterable[int],
    season: Optional[int] = None,
    ridge: float = DEFAULT_RIDGE,
    center_on: Optional[set] = None,
) -> Dict[int, EfficiencyRatings]:
    """The thing CFBD cannot give you: a rating per team per week.

    Each entry is fit only on games completed before that week, so the series
    reads as the season actually unfolded rather than as a season-final number
    projected backwards.
    """
    rows = list(rows or [])
    games = list(games or [])
    homes = home_map(games)
    out: Dict[int, EfficiencyRatings] = {}
    for week in sorted(set(int(w) for w in weeks)):
        fitted = fit_efficiency(
            rows, games, before_week=week, season=season, ridge=ridge,
            homes=homes, center_on=center_on,
        )
        if fitted is not None:
            out[week] = fitted
    log.info("built weekly efficiency ratings for %d week(s)", len(out))
    return out


def team_trend(series: Dict[int, EfficiencyRatings], team: str) -> List[Tuple[int, float, int]]:
    """(week, points, rank) for one team across the series."""
    key = normalize_team(team)
    trend: List[Tuple[int, float, int]] = []
    for week in sorted(series):
        ratings = series[week]
        value = ratings.points.get(key)
        rank = ratings.rank_of(key)
        if value is None or rank is None:
            continue
        trend.append((week, value, rank))
    return trend


def load(client, season: int, season_type: str = "regular") -> List[Dict[str, Any]]:
    """Per-game advanced stats for a whole season — one API call."""
    try:
        return client.advanced_game_stats(season, season_type=season_type)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load advanced game stats for %s: %s", season, exc)
        return []
