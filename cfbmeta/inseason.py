"""Point-in-time ratings, reconstructed from results available before kickoff.

This exists to answer the question the archive would otherwise take a full
season to answer: *given only what was knowable on the Thursday before a game,
does this model beat the closing line?*

CFBD can't answer it — its ratings endpoints are point-in-time and overwrite
themselves. But the raw material for a rating is not. Game results carry dates,
so the set of games completed before any past week is perfectly recoverable,
and a rating can be refit from exactly that set. Nothing about week 9 leaks
into a week-9 projection because the fit never sees a week-9 game.

The rating itself is a ridge-regularised least squares, which is what SRS is
underneath:

    margin = rating_home - rating_away + hfa

solved across every completed game, with a penalty pulling each team toward a
preseason prior (the previous season's final SP+, which genuinely existed).
Early in the season the penalty dominates and teams sit near their prior; as
games accumulate the results take over. That is the same shape as any real
in-season rating, and unlike the published ones it is reproducible for any week
of any past season.

It is not SP+. It has no play-by-play, no opponent-adjusted efficiency, no
garbage-time filtering. It is a fair *lower bound* on what a point-in-time
rating provides — which makes a backtest against it conservative rather than
flattering.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Strength of the pull toward the preseason prior, in units of "equivalent
# games". At 8, a team needs roughly eight results before the season's evidence
# outweighs where it started.
DEFAULT_RIDGE = 8.0


def completed_games(
    games: Iterable[dict], before_week: int, season: Optional[int] = None
) -> List[dict]:
    """Games finished strictly before ``before_week``.

    The strictness is the whole point: including the target week would leak the
    result being predicted back into the ratings predicting it.
    """
    out = []
    for game in games:
        week = pick_float(game, "week")
        home = pick_float(game, "homePoints", "home_points")
        away = pick_float(game, "awayPoints", "away_points")
        if week is None or home is None or away is None:
            continue
        if int(week) >= int(before_week):
            continue
        if season is not None and int(pick_float(game, "season", default=season)) != season:
            continue
        out.append(game)
    return out


def fit_ratings(
    games: Sequence[dict],
    prior: Optional[Dict[str, float]] = None,
    hfa: float = 2.7,
    ridge: float = DEFAULT_RIDGE,
    eligible: Optional[set] = None,
) -> Dict[str, float]:
    """Ridge-regularised margin ratings from a set of completed games.

    Returns team key -> rating in points above average. Teams with no games
    fall back to their prior (or zero).
    """
    import numpy as np

    prior = prior or {}
    rows: List[Tuple[str, str, float]] = []
    teams: List[str] = []
    index: Dict[str, int] = {}

    def slot(name: str) -> int:
        if name not in index:
            index[name] = len(teams)
            teams.append(name)
        return index[name]

    for game in games:
        home = normalize_team(pick(game, "homeTeam", "home_team") or "")
        away = normalize_team(pick(game, "awayTeam", "away_team") or "")
        hp = pick_float(game, "homePoints", "home_points")
        ap = pick_float(game, "awayPoints", "away_points")
        if not home or not away or hp is None or ap is None:
            continue
        if eligible is not None and (home not in eligible or away not in eligible):
            continue
        neutral = bool(pick(game, "neutralSite", "neutral_site", default=False))
        rows.append((home, away, (hp - ap) - (0.0 if neutral else hfa)))
        slot(home)
        slot(away)

    if not rows:
        return dict(prior)

    n_teams = len(teams)
    design = np.zeros((len(rows) + n_teams, n_teams))
    target = np.zeros(len(rows) + n_teams)

    for i, (home, away, margin) in enumerate(rows):
        design[i, index[home]] = 1.0
        design[i, index[away]] = -1.0
        target[i] = margin

    # Ridge block: sqrt(ridge) * (rating_t - prior_t) ~ 0. With no prior this
    # shrinks toward the mean, which is the right default for an unknown team.
    scale = float(ridge) ** 0.5
    for t, name in enumerate(teams):
        design[len(rows) + t, t] = scale
        target[len(rows) + t] = scale * prior.get(name, 0.0)

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    # Re-centre on zero so the scale matches the published ratings, which are
    # quoted against an average team.
    ratings = {name: float(solution[i]) for i, name in enumerate(teams)}
    mean = sum(ratings.values()) / len(ratings)
    ratings = {k: v - mean for k, v in ratings.items()}

    for name, value in prior.items():
        ratings.setdefault(name, value)
    return ratings


def weekly_ratings(
    games: Sequence[dict],
    weeks: Iterable[int],
    prior: Optional[Dict[str, float]] = None,
    hfa: float = 2.7,
    ridge: float = DEFAULT_RIDGE,
    eligible: Optional[set] = None,
) -> Dict[int, Dict[str, float]]:
    """Ratings as they would have stood entering each requested week."""
    out: Dict[int, Dict[str, float]] = {}
    for week in sorted(set(int(w) for w in weeks)):
        history = completed_games(games, before_week=week)
        out[week] = fit_ratings(history, prior=prior, hfa=hfa, ridge=ridge, eligible=eligible)
        log.debug("week %d fit on %d completed games", week, len(history))
    return out
