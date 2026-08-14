"""The meta-forecast: blend the component models, then adjust.

The pipeline for one game:

1. Each rating source proposes a neutral-field margin, ``home - away``, in
   points. Sources missing a rating for either team simply drop out and the
   weights renormalize over whatever is left.
2. Those are blended by weight into a neutral-field consensus.
3. Home field advantage is added once, on top — not per source, since every
   source is quoted as a neutral-field rating.
4. Situational adjustments (coaching, rest, travel) are added, each capped
   individually and capped again in aggregate.
5. The result is compared against the market line to produce an edge.

Nothing here invents a rating. The value is in the distillation, the explicit
adjustment ledger, and the honest uncertainty around the number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import Config
from .probability import (
    BetEvaluation,
    MarginDistribution,
    evaluate_spread_bet,
    evaluate_total_bet,
)
from .ratings import ALL_SOURCES, RatingBook, normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

SOURCE_LABELS = {
    "sp_plus": "SP+",
    "fpi": "FPI",
    "elo": "Elo",
    "srs": "SRS",
    "talent": "Talent",
    "market": "Market",
}

# Sources that only reflect results so far this season, so they carry very
# little signal in September and get downweighted early.
IN_SEASON_SOURCES = ("elo", "srs")
DEFAULT_POINTS_PER_TEAM = 27.5
# Below this many rated teams, a "league average" is not worth computing.
MIN_TEAMS_FOR_LEAGUE_STATS = 10


@dataclass
class ComponentMargin:
    source: str
    label: str
    margin: float  # neutral field, home perspective
    weight: float


@dataclass
class GameProjection:
    game_id: Any
    week: int
    season: int
    home_team: str
    away_team: str
    kickoff: Optional[datetime]
    neutral_site: bool = False
    venue: str = ""
    conference_game: bool = False

    components: List[ComponentMargin] = field(default_factory=list)
    neutral_margin: float = 0.0
    hfa: Dict[str, float] = field(default_factory=dict)
    coaching: Dict[str, Any] = field(default_factory=dict)
    situational: Dict[str, Any] = field(default_factory=dict)
    adjustment_total: float = 0.0

    projected_margin: float = 0.0
    projected_total: Optional[float] = None
    projected_home_points: Optional[float] = None
    projected_away_points: Optional[float] = None

    market_spread: Optional[float] = None
    market_total: Optional[float] = None
    market_provider: str = ""

    home_win_probability: float = 0.5
    spread_bet: Optional[BetEvaluation] = None
    total_bet: Optional[BetEvaluation] = None
    notes: List[str] = field(default_factory=list)

    @property
    def market_margin(self) -> Optional[float]:
        return None if self.market_spread is None else -self.market_spread

    @property
    def edge(self) -> Optional[float]:
        mm = self.market_margin
        return None if mm is None else round(self.projected_margin - mm, 2)

    @property
    def matchup(self) -> str:
        joiner = "vs" if self.neutral_site else "at"
        return f"{self.away_team} {joiner} {self.home_team}"

    @property
    def favorite(self) -> str:
        return self.home_team if self.projected_margin >= 0 else self.away_team

    @property
    def projected_line(self) -> str:
        """Our number expressed the way a book would quote it."""
        margin = abs(self.projected_margin)
        return f"{self.favorite} -{margin:.1f}"

    @property
    def confidence(self) -> str:
        n = len([c for c in self.components if c.source != "market"])
        if n >= 4 and self.market_spread is not None:
            return "high"
        if n >= 3:
            return "medium"
        return "low"

    def sort_key(self) -> float:
        if self.spread_bet and self.spread_bet.is_play:
            return abs(self.spread_bet.edge_points)
        return abs(self.edge or 0.0)


def effective_weights(config: Config, week: int) -> Dict[str, float]:
    """Blend weights for a given week.

    Elo and SRS describe only what has happened so far, which in week 2 is
    almost nothing. Early in the season their weight is shifted to the sources
    that carry a real preseason prior (SP+, FPI, talent). ``prior_weight_by_week``
    controls how much moves.
    """
    weights = config.weights.normalized()
    shift = config.prior_weight(week)
    if shift <= 0:
        return weights

    moved = 0.0
    for source in IN_SEASON_SOURCES:
        if weights.get(source):
            take = weights[source] * shift
            weights[source] -= take
            moved += take

    if moved > 0:
        targets = {s: weights.get(s, 0.0) for s in ("sp_plus", "fpi", "talent")}
        total = sum(targets.values())
        if total > 0:
            for source, current in targets.items():
                weights[source] = current + moved * (current / total)
        else:  # nothing to receive it; hand it back
            for source in IN_SEASON_SOURCES:
                if weights.get(source) is not None:
                    weights[source] += moved / len(IN_SEASON_SOURCES)
    return weights


def component_margins(
    book: RatingBook,
    home_team: str,
    away_team: str,
    weights: Dict[str, float],
    market_margin: Optional[float] = None,
) -> List[ComponentMargin]:
    """Neutral-field margin from each source that rates both teams."""
    out: List[ComponentMargin] = []
    for source in ALL_SOURCES:
        weight = weights.get(source, 0.0)
        if weight <= 0:
            continue
        home = book.rating(home_team, source)
        away = book.rating(away_team, source)
        if home is None or away is None:
            continue
        out.append(
            ComponentMargin(source, SOURCE_LABELS.get(source, source), round(home - away, 3), weight)
        )

    market_weight = weights.get("market", 0.0)
    if market_weight > 0 and market_margin is not None:
        out.append(ComponentMargin("market", "Market", round(market_margin, 3), market_weight))
    return out


def blend(components: List[ComponentMargin]) -> float:
    """Weighted mean, renormalized over the sources that actually showed up."""
    total_weight = sum(c.weight for c in components)
    if total_weight <= 0:
        return 0.0
    return sum(c.margin * c.weight for c in components) / total_weight


def detect_defense_sign(book: RatingBook) -> float:
    """Work out whether a higher SP+ defense rating is good or bad.

    SP+ has historically quoted defense as points allowed (lower is better),
    but rather than hardcode that, correlate defense with overall rating and
    read the sign off the data. Returns +1 when a higher defense rating means a
    better defense, -1 when it means a worse one.
    """
    pairs = [
        (t.meta["sp_defense"], t.values["sp_plus"])
        for t in book.teams.values()
        if "sp_defense" in t.meta and "sp_plus" in t.values
    ]
    if len(pairs) < MIN_TEAMS_FOR_LEAGUE_STATS:
        return -1.0
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return 1.0 if cov > 0 else -1.0


def project_points(
    book: RatingBook,
    home_team: str,
    away_team: str,
    defense_sign: float,
    base_points: float = DEFAULT_POINTS_PER_TEAM,
) -> Optional[Dict[str, float]]:
    """Score projection from SP+ offense/defense splits, if available."""
    offs = [t.meta["sp_offense"] for t in book.teams.values() if "sp_offense" in t.meta]
    defs = [t.meta["sp_defense"] for t in book.teams.values() if "sp_defense" in t.meta]
    if len(offs) < MIN_TEAMS_FOR_LEAGUE_STATS or len(defs) < MIN_TEAMS_FOR_LEAGUE_STATS:
        return None

    mean_off = sum(offs) / len(offs)
    mean_def = sum(defs) / len(defs)
    home, away = book.get(home_team), book.get(away_team)
    if not home or not away:
        return None
    for entry in (home, away):
        if "sp_offense" not in entry.meta or "sp_defense" not in entry.meta:
            return None

    def points_for(off_team, def_team) -> float:
        off_edge = off_team.meta["sp_offense"] - mean_off
        # defense_sign folds in whether higher = better defense.
        def_edge = -defense_sign * (def_team.meta["sp_defense"] - mean_def)
        return base_points + off_edge + def_edge

    hp = points_for(home, away)
    ap = points_for(away, home)
    return {"home_points": round(hp, 2), "away_points": round(ap, 2), "total": round(hp + ap, 2)}


def project_game(
    game: dict,
    book: RatingBook,
    config: Config,
    hfa_model=None,
    coach_model=None,
    situational=None,
    market: Optional[Dict[str, Any]] = None,
    defense_sign: float = -1.0,
) -> GameProjection:
    """Produce the full projection for a single game."""
    from .adjustments import parse_start

    home_team = pick(game, "homeTeam", "home_team", "home")
    away_team = pick(game, "awayTeam", "away_team", "away")
    week = int(pick_float(game, "week", default=1) or 1)
    neutral = bool(pick(game, "neutralSite", "neutral_site", default=False))
    kickoff = parse_start(pick(game, "startDate", "start_date", "startTime"))

    proj = GameProjection(
        game_id=pick(game, "id", "gameId"),
        week=week,
        season=int(pick_float(game, "season", "year", default=book.season) or book.season),
        home_team=home_team,
        away_team=away_team,
        kickoff=kickoff,
        neutral_site=neutral,
        venue=pick(game, "venue", "venueName", default="") or "",
        conference_game=bool(pick(game, "conferenceGame", "conference_game", default=False)),
    )

    market = market or {}
    proj.market_spread = market.get("spread")
    proj.market_total = market.get("total")
    proj.market_provider = market.get("provider", "")

    weights = effective_weights(config, week)
    proj.components = component_margins(
        book, home_team, away_team, weights, proj.market_margin
    )
    if not proj.components:
        proj.notes.append("No rating source covers both teams; no projection.")
        return proj

    proj.neutral_margin = round(blend(proj.components), 3)

    # -- home field ---------------------------------------------------------
    if hfa_model is not None:
        proj.hfa = hfa_model.for_game(home_team, away_team, neutral)
    else:
        proj.hfa = {"base": 0.0 if neutral else config.league_hfa, "elevation": 0.0,
                    "total": 0.0 if neutral else config.league_hfa}

    # -- adjustments --------------------------------------------------------
    adjustments = 0.0
    if coach_model is not None:
        proj.coaching = coach_model.for_game(home_team, away_team)
        adjustments += float(proj.coaching.get("total", 0.0))
    if situational is not None:
        proj.situational = situational.for_game(
            home_team,
            away_team,
            kickoff,
            venue_id=pick(game, "venueId", "venue_id"),
            neutral_site=neutral,
        )
        adjustments += float(proj.situational.get("total", 0.0))

    proj.adjustment_total = round(
        max(-config.total_adj_cap, min(config.total_adj_cap, adjustments)), 3
    )
    proj.projected_margin = round(
        proj.neutral_margin + proj.hfa.get("total", 0.0) + proj.adjustment_total, 2
    )

    # -- points / totals ----------------------------------------------------
    points = project_points(book, home_team, away_team, defense_sign)
    if points:
        # Re-center the score split on the margin we actually projected, so the
        # total and the spread can never disagree with each other.
        total = points["total"]
        proj.projected_total = round(total, 1)
        proj.projected_home_points = round((total + proj.projected_margin) / 2.0, 1)
        proj.projected_away_points = round((total - proj.projected_margin) / 2.0, 1)

    # -- probabilities and bets ---------------------------------------------
    dist = MarginDistribution(proj.projected_margin, config.sigma_margin)
    proj.home_win_probability = round(dist.p_home_win(), 4)

    proj.spread_bet = evaluate_spread_bet(
        proj.projected_margin,
        proj.market_spread,
        config.sigma_margin,
        price=config.vig_price,
        kelly_multiplier=config.kelly_fraction,
        bankroll_units=config.bankroll_units,
        min_edge_points=config.min_edge_points,
        max_units=config.max_units_per_play,
        edge_shrink=config.edge_shrink,
    )
    proj.total_bet = evaluate_total_bet(
        proj.projected_total,
        proj.market_total,
        config.sigma_total,
        price=config.vig_price,
        kelly_multiplier=config.kelly_fraction,
        bankroll_units=config.bankroll_units,
        max_units=config.max_units_per_play,
        edge_shrink=config.edge_shrink,
    )

    if proj.market_spread is None:
        proj.notes.append("No market line yet; edge cannot be graded.")
    missing = [
        SOURCE_LABELS[s]
        for s in ALL_SOURCES
        if weights.get(s, 0) > 0 and s not in {c.source for c in proj.components}
    ]
    if missing:
        proj.notes.append(f"Missing source(s): {', '.join(missing)}.")
    return proj


def clip_to_slate_window(
    games: List[dict], window_days: int = 6, now: Optional[datetime] = None
) -> List[dict]:
    """Keep one playing weekend, anchored on the next kickoff.

    CFBD weeks do not reliably correspond to a single weekend, so selecting a
    week is not enough to select a slate. The window starts at the earliest
    kickoff still ahead of ``now`` (falling back to the earliest kickoff at all,
    so previewing a past week still works) and runs ``window_days`` forward.
    """
    from .adjustments import parse_start

    now = now or datetime.now(timezone.utc)
    dated = []
    for game in games:
        kickoff = parse_start(pick(game, "startDate", "start_date", "startTime"))
        if kickoff is not None:
            dated.append((kickoff, game))
    if not dated:
        return games

    upcoming = [k for k, _ in dated if k >= now]
    anchor = min(upcoming) if upcoming else min(k for k, _ in dated)
    cutoff = anchor + timedelta(days=window_days)

    clipped = [g for k, g in dated if anchor <= k < cutoff]
    undated = [g for g in games if g not in [d[1] for d in dated]]
    if len(clipped) < len(games):
        log.info(
            "slate clipped to %s .. %s: %d of %d games",
            anchor.date(), cutoff.date(), len(clipped), len(games),
        )
    return clipped + undated


def project_slate(
    games: List[dict],
    book: RatingBook,
    config: Config,
    hfa_model=None,
    coach_model=None,
    situational=None,
    markets: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> List[GameProjection]:
    """Project every game on the slate, best edges first."""
    markets = markets or {}
    defense_sign = detect_defense_sign(book)
    projections: List[GameProjection] = []

    for game in games:
        home = pick(game, "homeTeam", "home_team", "home")
        away = pick(game, "awayTeam", "away_team", "away")
        if not home or not away:
            continue
        if not (book.has_ratings(home) and book.has_ratings(away)):
            # One side carries no usable rating — normally an FCS opponent, but
            # also every game in a season whose sources haven't published yet.
            # Either way there is nothing to project, and emitting a 0.0 margin
            # here would read as a pick'em rather than as missing data.
            log.debug("skipping %s at %s: no usable ratings", away, home)
            continue

        gid = pick(game, "id", "gameId")
        market = markets.get(gid) or markets.get(
            (normalize_team(home), normalize_team(away))
        )
        try:
            projections.append(
                project_game(
                    game, book, config, hfa_model, coach_model, situational, market, defense_sign
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad game shouldn't kill the slate
            log.warning("could not project %s at %s: %s", away, home, exc)

    projections.sort(key=lambda p: p.sort_key(), reverse=True)
    return projections
