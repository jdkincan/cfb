"""Slate-level staking: correlation, exposure caps, and manual resizing.

Kelly sizes one bet against one bankroll and assumes the next bet is
independent. A college football slate is not independent. Fifty games share a
weekend, a dozen share a conference, and — most importantly — they all share
*this model*. If the ratings are stale or a systematic bias has crept in, every
position on the card is wrong in the same direction at the same time.

So the per-bet Kelly stake is treated as a ceiling rather than an answer, and
two haircuts are applied on top:

* **Correlation** — the effective number of independent bets is smaller than
  the count, so the whole card is scaled by ``sqrt(n_eff / n)``. Bets sharing a
  conference or a kickoff window count as partially the same bet.
* **Exposure** — a hard cap on total units at risk in one week, because no
  edge estimate is confident enough to justify an unbounded card.

``resize`` handles the practical case: the readout suggests twelve plays, you
decide to bet four, and the four should be re-sized against the bankroll you
are actually risking rather than left at their original fractions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)


@dataclass
class Position:
    key: str                 # game id or any stable identifier
    label: str               # human-readable, e.g. "Texas -7.5"
    units: float             # stake before portfolio adjustment
    conference: str = ""
    kickoff_bucket: str = ""  # e.g. "2026-09-05-afternoon"
    edge: float = 0.0
    cover_probability: float = 0.0

    def group(self) -> str:
        return f"{self.conference}|{self.kickoff_bucket}"


@dataclass
class PortfolioResult:
    positions: List[Position] = field(default_factory=list)
    raw_units: float = 0.0
    scaled_units: float = 0.0
    correlation_factor: float = 1.0
    exposure_factor: float = 1.0
    effective_bets: float = 0.0

    @property
    def total_factor(self) -> float:
        return self.correlation_factor * self.exposure_factor

    def describe(self) -> str:
        return (
            f"{len(self.positions)} plays, {self.raw_units:.2f}u before scaling -> "
            f"{self.scaled_units:.2f}u "
            f"(correlation x{self.correlation_factor:.2f}, "
            f"exposure x{self.exposure_factor:.2f}, "
            f"effective bets {self.effective_bets:.1f})"
        )


def effective_bet_count(positions: Sequence[Position], within_group_rho: float = 0.4) -> float:
    """How many genuinely independent bets a card is worth.

    Uses the standard variance-of-a-sum argument: with equal weights and
    average pairwise correlation rho inside a group, n correlated bets carry the
    risk of n / (1 + (n-1) * rho) independent ones. Groups are summed.
    """
    if not positions:
        return 0.0
    groups: Dict[str, int] = {}
    for position in positions:
        groups[position.group()] = groups.get(position.group(), 0) + 1

    effective = 0.0
    for count in groups.values():
        effective += count / (1.0 + (count - 1) * within_group_rho)
    return effective


def apply(
    positions: List[Position],
    max_weekly_units: float = 10.0,
    within_group_rho: float = 0.4,
    max_units_per_play: float = 3.0,
) -> PortfolioResult:
    """Scale a card for correlation and total exposure."""
    result = PortfolioResult(positions=list(positions))
    result.raw_units = sum(p.units for p in positions)
    if not positions or result.raw_units <= 0:
        return result

    n = len(positions)
    result.effective_bets = effective_bet_count(positions, within_group_rho)
    # sqrt because risk scales with the square root of independent count.
    result.correlation_factor = min(1.0, math.sqrt(result.effective_bets / n))

    after_correlation = result.raw_units * result.correlation_factor
    result.exposure_factor = (
        min(1.0, max_weekly_units / after_correlation) if after_correlation > 0 else 1.0
    )

    for position in positions:
        position.units = round(
            min(position.units * result.total_factor, max_units_per_play), 2
        )
    result.scaled_units = round(sum(p.units for p in positions), 2)
    log.info("portfolio: %s", result.describe())
    return result


def resize(
    positions: Sequence[Position],
    keep: Sequence[str],
    max_weekly_units: float = 10.0,
    max_units_per_play: float = 3.0,
    within_group_rho: float = 0.4,
) -> PortfolioResult:
    """Re-size a card down to a chosen subset.

    Dropping eight of twelve plays does not mean betting the remaining four at
    their original stakes: the card is less diversified, so each surviving bet
    carries more of the week's risk. It also does not mean scaling them up to
    refill the exposure budget — the edge on each is unchanged. Kelly fractions
    are preserved and the correlation and exposure haircuts are recomputed on
    the smaller, more concentrated card.
    """
    wanted = {str(k) for k in keep}
    subset = [
        Position(**{**vars(p), "units": p.units}) for p in positions if str(p.key) in wanted
    ]
    missing = wanted - {str(p.key) for p in subset}
    if missing:
        log.warning("resize: no such play(s): %s", ", ".join(sorted(missing)))
    return apply(
        subset,
        max_weekly_units=max_weekly_units,
        within_group_rho=within_group_rho,
        max_units_per_play=max_units_per_play,
    )


def positions_from_projections(projections: Sequence[Any]) -> List[Position]:
    """Build portfolio positions from the plays on a slate."""
    out: List[Position] = []
    for proj in projections:
        bet = getattr(proj, "spread_bet", None)
        if not bet or not bet.is_play or bet.units <= 0:
            continue
        kickoff = getattr(proj, "kickoff", None)
        bucket = kickoff.strftime("%Y-%m-%d") if kickoff else ""
        side_team = proj.home_team if bet.side == "home" else proj.away_team
        out.append(Position(
            key=str(proj.game_id),
            label=f"{side_team} {bet.line:+.1f}",
            units=bet.units,
            conference=getattr(proj, "conference_game", "") and "conf" or "",
            kickoff_bucket=bucket,
            edge=bet.edge_points,
            cover_probability=bet.cover_probability,
        ))
    return out
