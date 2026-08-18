"""Manual adjustments for injuries and quarterback status.

There is no free injury feed. CFBD has no injury endpoint at any tier, ESPN's
injuries API returns 403, and the paid feeds sharps use (Sportradar, Rotowire)
are commercial products. Scraping beat reporters is not a data pipeline.

So this is deliberately manual: a small YAML file you edit when you know
something the model doesn't. That matches how the edge actually works. The
market prices *known* injuries efficiently within minutes of the news, so the
value was never in having the injury — it was in having it first, which a
Thursday-morning batch job structurally cannot. What a manual override buys is
the ability to *stand down* from a bet the model likes for a reason the model
cannot see, which is the realistic use.

Magnitudes, from the published research and market behaviour:

* **Quarterback** is the only position that moves a line much. An elite starter
  to a backup is worth roughly 7-10 points; an average starter to a competent
  backup, 3-4. This is the number worth getting approximately right.
* **Every other position** tops out around 1-1.5 points, even for stars.
  Skill-position and offensive-line depth matters in aggregate rather than
  individually.
* **Aggregate absences** (several starters out) are worth less than the sum of
  their parts; the cap below reflects that.

Format (``availability.yml`` at the repo root)::

    2026:
      week: 3
      adjustments:
        - team: Arkansas
          points: -6.5
          reason: "starting QB out, backup is a true freshman"
        - team: LSU
          points: -1.0
          reason: "two OL starters questionable"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .ratings import normalize_team

log = logging.getLogger(__name__)

AVAILABILITY_PATH = Path(__file__).resolve().parent.parent / "availability.yml"

# Guardrails. A single manual entry should not be able to swing a game more
# than losing a starting quarterback plausibly does.
MAX_PER_TEAM = 10.0
# Rules of thumb, exposed so the file can document itself.
QB_ELITE_TO_BACKUP = -8.0
QB_AVERAGE_TO_BACKUP = -3.5
SKILL_STARTER = -1.0
LINE_STARTER = -0.75


@dataclass
class Adjustment:
    team: str
    points: float
    reason: str = ""

    @property
    def key(self) -> str:
        return normalize_team(self.team)


@dataclass
class AvailabilityModel:
    week: Optional[int] = None
    adjustments: Dict[str, Adjustment] = field(default_factory=dict)

    def points_for(self, team: str) -> float:
        entry = self.adjustments.get(normalize_team(team))
        if entry is None:
            return 0.0
        return max(-MAX_PER_TEAM, min(MAX_PER_TEAM, entry.points))

    def reason_for(self, team: str) -> str:
        entry = self.adjustments.get(normalize_team(team))
        return entry.reason if entry else ""

    def for_game(self, home_team: str, away_team: str) -> Dict[str, object]:
        home = self.points_for(home_team)
        away = self.points_for(away_team)
        return {
            "home": round(home, 2),
            "away": round(away, 2),
            "total": round(home - away, 2),
            "home_reason": self.reason_for(home_team),
            "away_reason": self.reason_for(away_team),
        }

    def applies_to(self, week: int) -> bool:
        """Entries are week-scoped so last week's injuries don't linger."""
        return self.week is None or int(self.week) == int(week)


def load(
    season: int, week: Optional[int] = None, path: Optional[Path] = None
) -> AvailabilityModel:
    """Read manual adjustments for a season, if any are on file."""
    path = path or AVAILABILITY_PATH
    model = AvailabilityModel()
    if not path.exists():
        return model

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return model

    block = data.get(season) or data.get(str(season)) or {}
    if not isinstance(block, dict):
        return model

    model.week = block.get("week")
    if week is not None and not model.applies_to(week):
        log.info(
            "availability.yml targets week %s, not %s — ignoring stale entries",
            model.week, week,
        )
        return AvailabilityModel(week=model.week)

    for entry in block.get("adjustments") or []:
        team = entry.get("team")
        points = entry.get("points")
        if not team or points is None:
            continue
        adjustment = Adjustment(
            team=team, points=float(points), reason=entry.get("reason", "")
        )
        model.adjustments[adjustment.key] = adjustment

    if model.adjustments:
        log.info(
            "availability: %d manual adjustment(s) for week %s",
            len(model.adjustments), model.week,
        )
    return model
