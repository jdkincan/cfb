"""Roster talent built from recruiting classes, independent of /talent.

CFBD publishes a ready-made talent composite, but it lags: as of mid-August
2026 it has no rows for the coming season at all. Recruiting data does not lag,
because signing day is in February. So this rebuilds the same idea from the
underlying classes, which has the side benefit that the recency weighting is
ours to choose rather than a black box.

Two measures come out:

* **Talent score** — recency-weighted recruiting points across the last four
  classes. This is the direct stand-in for the talent composite and feeds the
  blend as the ``talent`` source.
* **Blue-chip ratio** — the share of those signees who were four- or five-star
  recruits. It is reported rather than blended: it is a roster-ceiling
  indicator (the shorthand being that essentially no team wins a title below
  ~50%), and it correlates far too tightly with the talent score to earn its
  own weight alongside it.

A caveat worth keeping in view: in the transfer-portal era, high school
recruiting explains less of a roster than it used to. A team can restock
through the portal in a single offseason and this measure will not see it.
Treat it as a prior on roster quality, which is exactly the weight it carries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..ratings import normalize_team
from .cfbd import pick, pick_float

log = logging.getLogger(__name__)

# How much each class contributes to the current roster. The incoming class is
# mostly freshmen who play sparingly; the classes two and three years back are
# the upperclassmen carrying the team; five years back has largely graduated.
CLASS_WEIGHTS = {0: 0.55, 1: 1.00, 2: 1.00, 3: 0.85}
BLUE_CHIP_STARS = 4


@dataclass
class TeamRecruiting:
    team: str
    points: float = 0.0  # recency-weighted class points
    signees: int = 0
    blue_chips: int = 0
    classes_seen: int = 0
    star_counts: Dict[int, int] = field(default_factory=dict)

    @property
    def blue_chip_ratio(self) -> Optional[float]:
        if self.signees < 20:  # too few signees to be a meaningful share
            return None
        return self.blue_chips / self.signees


def build_recruiting_profiles(
    client,
    season: int,
    classes: int = 4,
) -> Dict[str, TeamRecruiting]:
    """Aggregate the last ``classes`` recruiting classes into per-team profiles."""
    profiles: Dict[str, TeamRecruiting] = {}

    for offset in range(classes):
        year = season - offset
        weight = CLASS_WEIGHTS.get(offset, 0.5)

        try:
            team_rows = client.get("recruiting_teams", year=year)
        except Exception as exc:  # noqa: BLE001
            log.warning("no team recruiting for %d: %s", year, exc)
            team_rows = []
        for row in team_rows:
            team = pick(row, "team", "school")
            points = pick_float(row, "points")
            if not team or points is None:
                continue
            prof = profiles.setdefault(normalize_team(team), TeamRecruiting(team=team))
            prof.points += points * weight
            prof.classes_seen += 1

        try:
            player_rows = client.get("recruiting_players", year=year)
        except Exception as exc:  # noqa: BLE001
            log.warning("no player recruiting for %d: %s", year, exc)
            continue
        for row in player_rows:
            team = pick(row, "committedTo", "committed_to")
            stars = pick_float(row, "stars")
            if not team or stars is None:
                continue
            prof = profiles.setdefault(normalize_team(team), TeamRecruiting(team=team))
            star = int(stars)
            prof.signees += 1
            prof.star_counts[star] = prof.star_counts.get(star, 0) + 1
            if star >= BLUE_CHIP_STARS:
                prof.blue_chips += 1

    log.info(
        "recruiting: %d teams across %d classes (%d-%d)",
        len(profiles), classes, season - classes + 1, season,
    )
    return profiles


def talent_rows(profiles: Dict[str, TeamRecruiting]) -> List[Dict[str, object]]:
    """Shape recruiting profiles like the /talent payload the book expects."""
    return [
        {"team": prof.team, "talent": prof.points}
        for prof in profiles.values()
        if prof.points > 0
    ]


def blue_chip_table(profiles: Dict[str, TeamRecruiting], top: int = 25) -> List[tuple]:
    """(team, blue-chip ratio, signees) sorted best first — for reporting."""
    rows = [
        (p.team, p.blue_chip_ratio, p.signees)
        for p in profiles.values()
        if p.blue_chip_ratio is not None
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top]
