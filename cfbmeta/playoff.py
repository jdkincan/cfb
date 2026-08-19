"""Twelve-team playoff simulation: bids, seeds, bracket, title odds.

Format modelled here — the one in use from 2025 on, and the one asked for:

* Five conference champions get automatic bids, taken in committee-rank order.
* Seven at-large bids fill the field, again in committee-rank order.
* Seeding is **straight**: the twelve are seeded 1-12 by committee rank, so the
  top four seeds — and therefore the four first-round byes — are simply the
  four highest-ranked teams in the field, champion or not. (The 2024 edition
  instead handed byes to the top four *champions*, which is why a 12-2 team
  once hosted a bye it had not earned. That is not what is modelled.)
* First round is 5v12, 6v11, 7v10, 8v9 at the higher seed's home field.
  Quarterfinals onward are neutral-site.

The committee is the hard part
------------------------------
Everything above is mechanical once you know the ranking, and the ranking is a
human committee, not a formula. Rather than invent weights, the model here is
**fit to twelve seasons of final CFP rankings** (2014-2025, 300 ranked teams),
regressing each team's rank — converted to a latent quality scale — on its
record, its rating and its schedule strength.

What it learned, expressed in rating points so it can be sanity-checked:

======================  ==========================
one win                 worth +4.7 rating points
one loss                worth -13.1 rating points
one point of schedule   worth +2.5 rating points
======================  ==========================

That losses cost roughly three times what wins pay is the single most
important thing about how this committee behaves, and it is not something a
hand-tuned formula would have guessed. Held out one season at a time, the
model reproduces the real ranking at Spearman 0.89, with 10.8 of the actual
top twelve and 3.4 of the actual top four in the right group.

It is still a model of a room full of people. It cannot see a quarterback
injury in November, and it has no opinion about brand names. Treat a team on
the bubble as genuinely uncertain rather than as 46.3% likely.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .ratings import normalize_team
from .simulate import CHUNK, DEFAULT_SIGMA_TEAM, FCS_WIN_PROBABILITY, build_schedule

log = logging.getLogger(__name__)

DEFAULT_SIMS = 20000
FIELD_SIZE = 12
AUTO_BIDS = 5
BYES = 4

# Fitted on final CFP committee rankings, 2014-2025. See the module docstring;
# the fitting script lives in the commit that introduced this file.
COMMITTEE_CONST = 0.7048
COMMITTEE_WINS = 0.0707
COMMITTEE_LOSSES = -0.1988
COMMITTEE_RATING = 0.0151
COMMITTEE_SOS = 0.0384

# "reached at least this far". Bye teams skip a round but reach the
# quarterfinal, so the first entry is the field itself, not a game.
ROUNDS = ("field", "quarterfinal", "semifinal", "final", "champion")


@dataclass
class PlayoffOdds:
    """Everything one team's playoff picture looks like, as probabilities."""

    team: str
    display: str = ""
    conference: str = ""
    mean_wins: float = 0.0
    conference_title: float = 0.0
    auto_bid: float = 0.0
    at_large: float = 0.0
    make_field: float = 0.0
    bye: float = 0.0
    seed_counts: Dict[int, float] = field(default_factory=dict)
    reach: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_seed(self) -> Optional[float]:
        """Average seed *given* the team made the field."""
        total = sum(self.seed_counts.values())
        if total <= 0:
            return None
        return sum(s * c for s, c in self.seed_counts.items()) / total

    def title(self) -> float:
        return self.reach.get("champion", 0.0)


@dataclass
class PlayoffSimulation:
    season: int
    sims: int
    teams: Dict[str, PlayoffOdds] = field(default_factory=dict)
    games_simulated: int = 0
    games_locked: int = 0
    # How often each exact seeded field came up, and each unordered field.
    bracket_counts: Counter = field(default_factory=Counter)
    field_counts: Counter = field(default_factory=Counter)

    def modal_bracket(self) -> Tuple[List[str], float]:
        """The single most common seeded field, and how often it happened.

        Worth knowing that this probability is tiny — there are more plausible
        brackets than simulations — so it is a curiosity, not a forecast. For
        something to actually look at, use :meth:`chalk_bracket`.
        """
        if not self.bracket_counts:
            return [], 0.0
        seeded, count = self.bracket_counts.most_common(1)[0]
        return list(seeded), count / self.sims

    def modal_field(self) -> Tuple[FrozenSet[str], float]:
        """The most common set of twelve, ignoring how they were seeded."""
        if not self.field_counts:
            return frozenset(), 0.0
        members, count = self.field_counts.most_common(1)[0]
        return members, count / self.sims

    def chalk_bracket(self) -> List[Tuple[int, str, float]]:
        """The likeliest occupant of each seed line, one team per line.

        Taking the argmax of each seed independently would put the same team on
        several lines, so this solves the assignment instead: pick the set of
        twelve team-to-seed pairings that maximises total probability. Filling
        greedily from seed 1 down is not the same thing — it can strand a line
        with a leftover after better candidates are spent — so the greedy pass
        is followed by pairwise swaps until no swap improves the total.

        Returns (seed, team, probability that team lands on exactly that seed).
        """
        candidates = [k for k, o in self.teams.items() if o.seed_counts]
        if not candidates:
            return []

        def p(team: str, seed: int) -> float:
            return self.teams[team].seed_counts.get(seed, 0.0)

        seeds = list(range(1, FIELD_SIZE + 1))
        placed: set = set()
        assign: Dict[int, str] = {}
        for seed in seeds:
            pool = [k for k in candidates if k not in placed]
            if not pool:
                break
            best = max(pool, key=lambda k: p(k, seed))
            assign[seed] = best
            placed.add(best)

        # Swap improvement: both between two assigned lines, and between an
        # assigned line and an unused team.
        improved = True
        while improved:
            improved = False
            for i in assign:
                for j in assign:
                    if i >= j:
                        continue
                    a, b = assign[i], assign[j]
                    if p(a, i) + p(b, j) < p(b, i) + p(a, j) - 1e-12:
                        assign[i], assign[j] = b, a
                        improved = True
            bench = [k for k in candidates if k not in set(assign.values())]
            for seed in assign:
                current = assign[seed]
                for other in bench:
                    if p(other, seed) > p(current, seed) + 1e-12:
                        assign[seed] = other
                        current = other
                        improved = True
        return [(seed, assign[seed], p(assign[seed], seed))
                for seed in sorted(assign)]

    def ranked(self, by: str = "make_field", limit: Optional[int] = None) -> List[PlayoffOdds]:
        key = (lambda t: -t.title()) if by == "title" else (lambda t: -getattr(t, by))
        rows = sorted(self.teams.values(), key=key)
        return rows[:limit] if limit else rows


def committee_score(
    wins: float, losses: float, rating: float, sos: float
) -> float:
    """The fitted committee model, for one team, on the latent quality scale."""
    return (
        COMMITTEE_CONST
        + COMMITTEE_WINS * wins
        + COMMITTEE_LOSSES * losses
        + COMMITTEE_RATING * rating
        + COMMITTEE_SOS * sos
    )


def strength_of_schedule(
    schedule: Sequence[dict], names: Sequence[str], ratings: Sequence[float]
) -> "Any":
    """Mean opponent rating per team.

    Fixed across simulations: it depends on who you played, not on how the
    coin landed. Computing it once keeps it out of the inner loop.
    """
    import numpy as np

    total = np.zeros(len(names))
    count = np.zeros(len(names))
    for game in schedule:
        h, a = game["home"], game["away"]
        total[h] += ratings[a]
        count[h] += 1
        total[a] += ratings[h]
        count[a] += 1
    return np.divide(total, np.maximum(count, 1))


def _bracket_pairs(field_idx: Sequence[int]) -> List[Tuple[int, int]]:
    """First-round matchups for a straight-seeded twelve-team field.

    ``field_idx`` is in seed order, seed 1 first. Seeds 1-4 are absent from
    the result: they are on bye.
    """
    return [
        (field_idx[4], field_idx[11]),   # 5 v 12
        (field_idx[5], field_idx[10]),   # 6 v 11
        (field_idx[6], field_idx[9]),    # 7 v 10
        (field_idx[7], field_idx[8]),    # 8 v 9
    ]


def simulate_playoff(
    games: Iterable[dict],
    base_margin: Callable[[dict], Optional[float]],
    team_rating: Dict[str, float],
    season: int,
    sims: int = DEFAULT_SIMS,
    sigma_game: float = 16.0,
    sigma_team: float = DEFAULT_SIGMA_TEAM,
    playoff_hfa: float = 2.7,
    eligible: Optional[set] = None,
    seed: Optional[int] = 0,
) -> PlayoffSimulation:
    """Play the season, the conference championships and the bracket, ``sims`` times.

    The team-strength offset drawn for the regular season is carried into the
    playoff games in the same simulated universe. That matters: a team that
    over-performed its rating all year because the rating was wrong should keep
    over-performing in January, and a bracket run off fresh draws would wash
    that out and understate how often a hot team runs the table.
    """
    import numpy as np

    # Team keys inside the simulation are normalized, so normalize what the
    # caller handed us rather than silently rating every team 0.0 because it
    # passed display names.
    team_rating = {normalize_team(k): v for k, v in (team_rating or {}).items()}
    eligible = {normalize_team(k) for k in eligible} if eligible is not None else None

    schedule, index, names, display = build_schedule(games, base_margin, eligible)
    if not schedule:
        return PlayoffSimulation(season=season, sims=0)

    n = len(names)
    ratings = np.array([team_rating.get(k, 0.0) for k in names], dtype=float)
    rated = np.array(
        [(eligible is None or k in eligible) and k in team_rating for k in names]
    )
    if not rated.any():
        log.warning("no team carries both a rating and eligibility; nothing to simulate")
        return PlayoffSimulation(season=season, sims=0)
    sos = strength_of_schedule(schedule, names, ratings)

    played = [g for g in schedule if g["outcome"] is not None]
    unplayed = [g for g in schedule if g["outcome"] is None and g["fcs_side"] is None]
    fcs = [g for g in schedule if g["outcome"] is None and g["fcs_side"] is not None]

    base_wins = np.zeros(n)
    base_conf = np.zeros(n)
    scheduled = np.zeros(n)
    conf_games = np.zeros(n)
    for g in schedule:
        scheduled[g["home"]] += 1
        scheduled[g["away"]] += 1
        if g["conference_game"]:
            conf_games[g["home"]] += 1
            conf_games[g["away"]] += 1
    for g in played:
        w = g["home"] if g["outcome"] else g["away"]
        base_wins[w] += 1
        if g["conference_game"]:
            base_conf[w] += 1

    home_idx = np.array([g["home"] for g in unplayed], dtype=np.int64)
    away_idx = np.array([g["away"] for g in unplayed], dtype=np.int64)
    margins = np.array([g["margin"] for g in unplayed], dtype=float)
    is_conf = np.array([g["conference_game"] for g in unplayed], dtype=bool)
    fcs_win = np.array([g[g["fcs_side"]] for g in fcs], dtype=np.int64)
    fcs_lose = np.array(
        [g["away" if g["fcs_side"] == "home" else "home"] for g in fcs], dtype=np.int64
    )

    # Conference membership, restricted to leagues that actually play a
    # conference schedule — independents cannot win a title they never contest.
    conference_of: Dict[str, str] = {}
    for g in schedule:
        for side, conf in (("home", "home_conf"), ("away", "away_conf")):
            key = names[g[side]]
            if g[conf] and key not in conference_of:
                conference_of[key] = g[conf]
    groups: Dict[str, List[int]] = {}
    for i, key in enumerate(names):
        conf = conference_of.get(key, "")
        if not conf or not rated[i] or conf_games[i] <= 0:
            continue
        groups.setdefault(conf, []).append(i)
    groups = {c: m for c, m in groups.items() if len(m) >= 2}

    rng = np.random.default_rng(seed)
    static = COMMITTEE_CONST + COMMITTEE_RATING * ratings + COMMITTEE_SOS * sos
    eligible_idx = np.where(rated)[0]

    bracket_counts: Counter = Counter()
    field_counts: Counter = Counter()
    title_ct = np.zeros(n)
    auto_ct = np.zeros(n)
    large_ct = np.zeros(n)
    bye_ct = np.zeros(n)
    seed_ct = np.zeros((n, FIELD_SIZE + 1))
    reach_ct = {r: np.zeros(n) for r in ROUNDS}
    wins_total = np.zeros(n)

    def play(a: int, b: int, off, hfa: float) -> int:
        """One neutral-or-home game inside a single simulated universe."""
        margin = ratings[a] - ratings[b] + off[a] - off[b] + hfa
        return a if margin + rng.normal(0.0, sigma_game) > 0 else b

    done = 0
    while done < sims:
        batch = min(CHUNK, sims - done)
        offsets = rng.normal(0.0, sigma_team, size=(batch, n))
        noise = rng.normal(0.0, sigma_game, size=(batch, len(unplayed)))
        realised = margins[None, :] + offsets[:, home_idx] - offsets[:, away_idx] + noise
        home_won = realised > 0.0

        wins = np.tile(base_wins, (batch, 1))
        cwins = np.tile(base_conf, (batch, 1))
        np.add.at(wins.T, home_idx, home_won.T.astype(float))
        np.add.at(wins.T, away_idx, (~home_won).T.astype(float))
        if len(fcs):
            fbs_won = rng.random((batch, len(fcs))) < FCS_WIN_PROBABILITY
            np.add.at(wins.T, fcs_win, fbs_won.T.astype(float))
            np.add.at(wins.T, fcs_lose, (~fbs_won).T.astype(float))
        if is_conf.any():
            np.add.at(cwins.T, home_idx, (home_won & is_conf[None, :]).T.astype(float))
            np.add.at(cwins.T, away_idx, ((~home_won) & is_conf[None, :]).T.astype(float))

        wins_total += wins.sum(axis=0)
        # Pre-championship committee score, used to break conference ties.
        pre = static[None, :] + COMMITTEE_WINS * wins + COMMITTEE_LOSSES * (
            scheduled[None, :] - wins
        )

        for s in range(batch):
            off = offsets[s]
            w = wins[s].copy()
            g = scheduled.copy()
            champions: List[int] = []

            # --- conference championship games ------------------------------
            for members in groups.values():
                order = sorted(
                    members, key=lambda i: (-cwins[s][i], -pre[s][i])
                )
                a, b = order[0], order[1]
                winner = play(a, b, off, 0.0)
                loser = b if winner is a else a
                w[winner] += 1
                g[a] += 1
                g[b] += 1
                champions.append(winner)
                del loser

            # --- committee ranking ------------------------------------------
            score = static + COMMITTEE_WINS * w + COMMITTEE_LOSSES * (g - w)
            order = eligible_idx[np.argsort(-score[eligible_idx])]

            # --- bids: five champions by rank, then at-large to twelve ------
            champ_set = set(champions)
            field_idx: List[int] = []
            autos: List[int] = []
            for i in order:
                if len(autos) >= AUTO_BIDS:
                    break
                if int(i) in champ_set:
                    autos.append(int(i))
            field_idx.extend(autos)
            auto_set = set(autos)
            for i in order:
                if len(field_idx) >= FIELD_SIZE:
                    break
                if int(i) not in auto_set:
                    field_idx.append(int(i))
            if len(field_idx) < FIELD_SIZE:
                continue

            # Straight seeding: reorder the twelve by committee rank.
            field_idx.sort(key=lambda i: -score[i])

            seeded_keys = tuple(names[i] for i in field_idx)
            bracket_counts[seeded_keys] += 1
            field_counts[frozenset(seeded_keys)] += 1
            for pos, i in enumerate(field_idx, 1):
                seed_ct[i, pos] += 1
                if pos <= BYES:
                    bye_ct[i] += 1
            for i in autos:
                auto_ct[i] += 1
            for i in field_idx:
                if i not in auto_set:
                    large_ct[i] += 1
            for i in champions:
                title_ct[i] += 1
            for i in field_idx:
                reach_ct["field"][i] += 1

            # --- bracket ----------------------------------------------------
            r1 = [play(hi, lo, off, playoff_hfa) for hi, lo in _bracket_pairs(field_idx)]
            # Quarterfinals, neutral: 1 v W(8/9), 2 v W(7/10), 3 v W(6/11), 4 v W(5/12).
            qf_pairs = [
                (field_idx[0], r1[3]),
                (field_idx[3], r1[0]),
                (field_idx[2], r1[1]),
                (field_idx[1], r1[2]),
            ]
            for a, b in qf_pairs:
                reach_ct["quarterfinal"][a] += 1
                reach_ct["quarterfinal"][b] += 1
            qf = [play(a, b, off, 0.0) for a, b in qf_pairs]
            # Semifinals keep 1 and 2 on opposite sides of the draw.
            sf_pairs = [(qf[0], qf[1]), (qf[3], qf[2])]
            for a, b in sf_pairs:
                reach_ct["semifinal"][a] += 1
                reach_ct["semifinal"][b] += 1
            sf = [play(a, b, off, 0.0) for a, b in sf_pairs]
            reach_ct["final"][sf[0]] += 1
            reach_ct["final"][sf[1]] += 1
            reach_ct["champion"][play(sf[0], sf[1], off, 0.0)] += 1

        done += batch

    result = PlayoffSimulation(
        season=season, sims=sims,
        games_simulated=len(unplayed) + len(fcs), games_locked=len(played),
        bracket_counts=bracket_counts, field_counts=field_counts,
    )
    for i, key in enumerate(names):
        if not rated[i]:
            continue
        seeds = {
            s: float(seed_ct[i, s] / sims) for s in range(1, FIELD_SIZE + 1)
            if seed_ct[i, s] > 0
        }
        result.teams[key] = PlayoffOdds(
            team=key, display=display.get(key, key),
            conference=conference_of.get(key, ""),
            mean_wins=float(wins_total[i] / sims),
            conference_title=float(title_ct[i] / sims),
            auto_bid=float(auto_ct[i] / sims),
            at_large=float(large_ct[i] / sims),
            make_field=float(sum(seed_ct[i, 1:]) / sims),
            bye=float(bye_ct[i] / sims),
            seed_counts=seeds,
            reach={r: float(reach_ct[r][i] / sims) for r in ROUNDS},
        )
    log.info(
        "simulated %d playoffs: %d conferences award automatic bids, "
        "%d teams eligible", sims, len(groups), int(rated.sum()),
    )
    return result
