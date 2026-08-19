"""Monte Carlo the season: win totals, conference races, and how sure we are.

Why a season sim needs more than per-game noise
-----------------------------------------------
The naive version draws ``margin = projection + N(0, sigma)`` for each game
independently and counts wins. It produces win-total distributions that are far
too narrow, and it is wrong for a specific reason: it treats each team's rating
as *known*. It isn't. A preseason rating is an estimate, and if a team is
actually two points better than we think, that error applies to all twelve of
its games in the same direction. Independent game noise averages out over a
season; a wrong rating does not.

So each simulated season draws a per-team strength offset once, holds it for
that team's whole schedule, and layers game noise on top::

    margin = base_margin + offset[home] - offset[away] + N(0, sigma_game)

``sigma_team`` is the standard deviation of that offset. It is not guessed: it
is fit by requiring the simulator's prediction intervals to have the coverage
they claim on completed seasons — an 80% interval should contain the actual win
total about 80% of the time. See :func:`calibrate_sigma_team`, and note that
without the team offset the intervals are badly over-confident, which is the
failure mode that makes a season sim feel authoritative and be useless.

Partial seasons are supported: games with a final score are locked to what
actually happened and only the remainder is simulated, so this stays useful in
November.

What it does not do
-------------------
No playoff odds. A 12-team field with committee seeding, automatic bids and
tiebreakers is not something this can model honestly, and a made-up number
would be worse than no number. Conference titles are approximated as "most
conference wins, ties broken at random", which ignores championship games and
head-to-head rules — reported, but do not bet it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Per-team season-long strength error, fit by interval coverage over 2023-25
# through the real projection pipeline, under preseason conditions: every game
# rated from the prior season's final ratings, all results hidden.
#
#   sigma_team    80% covers    50% covers    95% covers
#          0.0         70.6%         46.4%             -
#          4.0         74.0%         49.9%         89.6%
#          6.0         78.7%         54.3%         94.0%   <-
#          8.0         82.9%         59.5%         96.3%
#         10.0         85.4%         62.0%         97.8%
#
# Zero is badly over-confident, which is the whole point of the parameter.
# 6.0 lands closest to nominal at all three levels at once. Mean-win MAE is
# 1.84 regardless, since the team offset widens the distribution without
# moving its centre.
#
# Conservative for a live run: fit using last season's final ratings, while a
# real preseason forecast also has SP+ and FPI, which already price returning
# production and recruiting. Live intervals should be a little tighter.
DEFAULT_SIGMA_TEAM = 6.0
DEFAULT_SIMS = 20000
# Chunk the Monte Carlo so a big run does not allocate sims x games at once.
CHUNK = 2000

# An FBS team hosting an FCS opponent cannot be projected — the rating book
# does not usually cover FCS — but dropping the game costs the FBS team most of
# a win, which is a large error on a 12-game season. Measured over 2022-25:
# FBS teams went 463-22 (95.5%) against FCS, by a mean of 31 points. So these
# are simulated as a fixed-probability win rather than discarded.
FCS_WIN_PROBABILITY = 0.955


@dataclass
class TeamOutcome:
    team: str
    display: str = ""
    conference: str = ""
    played: int = 0
    actual_wins: int = 0
    scheduled: int = 0
    mean_wins: float = 0.0
    win_counts: Dict[int, float] = field(default_factory=dict)
    mean_conference_wins: float = 0.0
    # None where the idea does not apply — independents play no conference
    # games, so "most conference wins" would be a coin flip between teams that
    # all finished 0-0.
    conference_title: Optional[float] = None

    def probability_of_at_least(self, wins: int) -> float:
        return sum(p for w, p in self.win_counts.items() if w >= wins)

    def interval(self, level: float = 0.80) -> Tuple[int, int]:
        """Central credible interval on the win total."""
        tail = (1.0 - level) / 2.0
        ordered = sorted(self.win_counts)
        cumulative = 0.0
        low = ordered[0] if ordered else 0
        high = ordered[-1] if ordered else 0
        for wins in ordered:
            cumulative += self.win_counts[wins]
            if cumulative >= tail:
                low = wins
                break
        cumulative = 0.0
        for wins in reversed(ordered):
            cumulative += self.win_counts[wins]
            if cumulative >= tail:
                high = wins
                break
        return low, max(high, low)


@dataclass
class SeasonSimulation:
    season: int
    sims: int
    teams: Dict[str, TeamOutcome] = field(default_factory=dict)
    games_simulated: int = 0
    games_locked: int = 0
    games_base_rate: int = 0
    sigma_game: float = 16.0
    sigma_team: float = DEFAULT_SIGMA_TEAM

    def ranked(self, by: str = "mean_wins", limit: Optional[int] = None) -> List[TeamOutcome]:
        rows = sorted(self.teams.values(), key=lambda t: -getattr(t, by))
        return rows[:limit] if limit else rows

    def conference(self, name: str) -> List[TeamOutcome]:
        key = name.lower()
        rows = [t for t in self.teams.values() if t.conference.lower() == key]
        return sorted(rows, key=lambda t: -t.mean_conference_wins)


def _result(game: dict) -> Optional[bool]:
    """True if the home team won, False if it lost, None if unplayed."""
    home = pick_float(game, "homePoints", "home_points")
    away = pick_float(game, "awayPoints", "away_points")
    if home is None or away is None:
        return None
    if home == away:
        return None
    return home > away


def build_schedule(
    games: Iterable[dict],
    base_margin: Callable[[dict], Optional[float]],
    eligible: Optional[set] = None,
) -> Tuple[List[dict], Dict[str, int], List[str], Dict[str, str]]:
    """Index the schedule for simulation.

    ``base_margin`` returns the projected home margin including home field and
    every adjustment — i.e. exactly what the weekly forecast would say — so the
    simulation and the Thursday readout cannot drift apart.
    """
    index: Dict[str, int] = {}
    names: List[str] = []
    prepared: List[dict] = []
    display: Dict[str, str] = {}

    def slot(key: str, label: str = "") -> int:
        if key not in index:
            index[key] = len(names)
            names.append(key)
        if label:
            display.setdefault(key, label)
        return index[key]

    for game in games or []:
        home = pick(game, "homeTeam", "home_team")
        away = pick(game, "awayTeam", "away_team")
        if not home or not away:
            continue
        hkey, akey = normalize_team(home), normalize_team(away)
        home_ok = eligible is None or hkey in eligible
        away_ok = eligible is None or akey in eligible
        if not home_ok and not away_ok:
            continue

        outcome = _result(game)
        margin = None
        fcs_side = None
        if outcome is None:
            margin = base_margin(game)
            if margin is None:
                if home_ok != away_ok:
                    # One rated side, one unrated: an FBS-vs-FCS game. Keep it
                    # at the historical rate instead of silently deleting a win.
                    fcs_side = "home" if home_ok else "away"
                else:
                    continue  # genuinely unprojectable

        prepared.append({
            "home": slot(hkey, home), "away": slot(akey, away),
            "home_name": home, "away_name": away,
            "margin": margin, "outcome": outcome, "fcs_side": fcs_side,
            "conference_game": bool(
                pick(game, "conferenceGame", "conference_game", default=False)
            ),
            "home_conf": pick(game, "homeConference", "home_conference", default="") or "",
            "away_conf": pick(game, "awayConference", "away_conference", default="") or "",
        })
    return prepared, index, names, display


def simulate(
    games: Iterable[dict],
    base_margin: Callable[[dict], Optional[float]],
    season: int,
    sims: int = DEFAULT_SIMS,
    sigma_game: float = 16.0,
    sigma_team: float = DEFAULT_SIGMA_TEAM,
    eligible: Optional[set] = None,
    seed: Optional[int] = 0,
) -> SeasonSimulation:
    """Run the season ``sims`` times and tally what happened."""
    import numpy as np

    schedule, index, names, display = build_schedule(games, base_margin, eligible)
    if not schedule:
        return SeasonSimulation(season=season, sims=0)

    n_teams = len(names)
    played = [g for g in schedule if g["outcome"] is not None]
    unplayed = [g for g in schedule
                if g["outcome"] is None and g["fcs_side"] is None]
    fcs = [g for g in schedule if g["outcome"] is None and g["fcs_side"] is not None]

    rng = np.random.default_rng(seed)

    # Locked-in results: no randomness, so tally them once.
    base_wins = np.zeros(n_teams)
    base_conf_wins = np.zeros(n_teams)
    conf_games = np.zeros(n_teams)
    scheduled = np.zeros(n_teams)
    for g in schedule:
        scheduled[g["home"]] += 1
        scheduled[g["away"]] += 1
        if g["conference_game"]:
            conf_games[g["home"]] += 1
            conf_games[g["away"]] += 1
    for g in played:
        winner = g["home"] if g["outcome"] else g["away"]
        base_wins[winner] += 1
        if g["conference_game"]:
            base_conf_wins[winner] += 1

    home_idx = np.array([g["home"] for g in unplayed], dtype=np.int64)
    away_idx = np.array([g["away"] for g in unplayed], dtype=np.int64)
    margins = np.array([g["margin"] for g in unplayed], dtype=float)
    is_conf = np.array([g["conference_game"] for g in unplayed], dtype=bool)
    # FBS side of each unprojectable FBS-vs-FCS game, and its opponent.
    fcs_win_idx = np.array(
        [g[g["fcs_side"]] for g in fcs], dtype=np.int64
    )
    fcs_lose_idx = np.array(
        [g["away" if g["fcs_side"] == "home" else "home"] for g in fcs],
        dtype=np.int64,
    )

    # Tally: win counts per team, and conference wins per sim for titles.
    win_hist = np.zeros((n_teams, int(scheduled.max()) + 1))
    conf_win_total = np.zeros(n_teams)
    title_count = np.zeros(n_teams)
    conference_of = _conference_map(schedule, names)
    conf_groups = _group_by_conference(conference_of, names)
    # A conference title needs conference games. Independents have none, so
    # every member finishes 0-0 and "most conference wins" degenerates into a
    # coin flip — a number that looks like a forecast and is pure noise.
    conf_groups = {
        conf: members for conf, members in conf_groups.items()
        if any(conf_games[i] > 0 for i in members)
    }

    done = 0
    while done < sims:
        batch = min(CHUNK, sims - done)
        # One strength offset per team per simulated season — held across that
        # team's whole schedule. This is the piece that makes the win-total
        # spread realistic rather than binomial.
        offsets = rng.normal(0.0, sigma_team, size=(batch, n_teams))
        noise = rng.normal(0.0, sigma_game, size=(batch, len(unplayed)))
        realised = (
            margins[None, :] + offsets[:, home_idx] - offsets[:, away_idx] + noise
        )
        home_won = realised > 0.0

        wins = np.tile(base_wins, (batch, 1))
        cwins = np.tile(base_conf_wins, (batch, 1))
        np.add.at(wins.T, home_idx, home_won.T.astype(float))
        np.add.at(wins.T, away_idx, (~home_won).T.astype(float))
        if len(fcs):
            fbs_won = rng.random((batch, len(fcs))) < FCS_WIN_PROBABILITY
            np.add.at(wins.T, fcs_win_idx, fbs_won.T.astype(float))
            np.add.at(wins.T, fcs_lose_idx, (~fbs_won).T.astype(float))
        if is_conf.any():
            ch = home_won & is_conf[None, :]
            ca = (~home_won) & is_conf[None, :]
            np.add.at(cwins.T, home_idx, ch.T.astype(float))
            np.add.at(cwins.T, away_idx, ca.T.astype(float))

        for team in range(n_teams):
            counts = np.bincount(wins[:, team].astype(int), minlength=win_hist.shape[1])
            win_hist[team, : len(counts)] += counts
        conf_win_total += cwins.sum(axis=0)

        for members in conf_groups.values():
            if len(members) < 2:
                continue
            block = cwins[:, members]
            # Random tiebreak: jitter smaller than any real win difference.
            jitter = rng.random(block.shape) * 1e-6
            winners = np.array(members)[np.argmax(block + jitter, axis=1)]
            np.add.at(title_count, winners, 1.0)

        done += batch

    result = SeasonSimulation(
        season=season, sims=sims, games_simulated=len(unplayed) + len(fcs),
        games_base_rate=len(fcs),
        games_locked=len(played), sigma_game=sigma_game, sigma_team=sigma_team,
    )
    for i, key in enumerate(names):
        if eligible is not None and key not in eligible:
            continue  # an FCS opponent that only exists as somebody's schedule
        hist = win_hist[i]
        total = hist.sum()
        distribution = {
            int(w): float(c / total) for w, c in enumerate(hist) if c > 0
        }
        conf = conference_of.get(key, "")
        result.teams[key] = TeamOutcome(
            team=key,
            display=display.get(key, key),
            conference=conf,
            played=int(sum(1 for g in played if key in (names[g["home"]], names[g["away"]]))),
            actual_wins=int(base_wins[i]),
            scheduled=int(scheduled[i]),
            mean_wins=float(sum(w * p for w, p in distribution.items())),
            win_counts=distribution,
            mean_conference_wins=float(conf_win_total[i] / sims),
            conference_title=(
                float(title_count[i] / sims) if conf in conf_groups else None
            ),
        )
    log.info(
        "simulated %d seasons: %d games projected, %d at the FBS-vs-FCS base "
        "rate, %d already final", sims, len(unplayed), len(fcs), len(played),
    )
    return result


def _conference_map(schedule: Sequence[dict], names: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for g in schedule:
        if g["home_conf"]:
            out.setdefault(names[g["home"]], g["home_conf"])
        if g["away_conf"]:
            out.setdefault(names[g["away"]], g["away_conf"])
    return out


def _group_by_conference(
    conference_of: Dict[str, str], names: Sequence[str]
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(names):
        conf = conference_of.get(key)
        if conf:
            groups.setdefault(conf, []).append(i)
    return groups


# -- calibration -------------------------------------------------------------
def coverage_report(
    simulation: SeasonSimulation,
    actual_wins: Dict[str, int],
    level: float = 0.80,
) -> Dict[str, float]:
    """How often the claimed interval actually contained the truth.

    The number that says whether ``sigma_team`` is honest. A simulator whose
    80% interval covers 55% of outcomes is not conservative, it is wrong, and
    every probability it reports is overstated.
    """
    inside = 0
    total = 0
    errors: List[float] = []
    for key, outcome in simulation.teams.items():
        truth = actual_wins.get(key)
        if truth is None:
            continue
        low, high = outcome.interval(level)
        total += 1
        if low <= truth <= high:
            inside += 1
        errors.append(abs(outcome.mean_wins - truth))
    if not total:
        return {"coverage": float("nan"), "claimed": level, "n": 0, "mae": float("nan")}
    return {
        "coverage": inside / total,
        "claimed": level,
        "n": total,
        "mae": sum(errors) / len(errors),
    }


def actual_win_totals(games: Iterable[dict], eligible: Optional[set] = None) -> Dict[str, int]:
    wins: Dict[str, int] = {}
    for game in games or []:
        outcome = _result(game)
        if outcome is None:
            continue
        home = normalize_team(pick(game, "homeTeam", "home_team", default="") or "")
        away = normalize_team(pick(game, "awayTeam", "away_team", default="") or "")
        if not home or not away:
            continue
        # Count a win for any eligible team, including against an opponent
        # outside the pool — the simulator now models those games too, so
        # filtering them here would compare different quantities.
        for team in (home, away):
            if eligible is None or team in eligible:
                wins.setdefault(team, 0)
        winner = home if outcome else away
        if winner in wins:
            wins[winner] += 1
    return wins
