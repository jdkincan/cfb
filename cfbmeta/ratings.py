"""Normalize every rating source onto one common scale: points above average.

SP+, FPI and SRS are already published as net points per game against an
average opponent, so they pass through untouched. Elo and the talent composite
live on their own scales, so they are z-scored across FBS and then stretched to
the same standard deviation the SP+ ratings show that season. That keeps the
mapping self-calibrating instead of relying on a hardcoded conversion constant.

With every source in points, each one projects a margin the same way:

    margin = home_rating - away_rating + home_field_advantage
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .sources.cfbd import pick, pick_float

# Sources already denominated in points per game.
NATIVE_POINT_SOURCES = {"sp_plus", "fpi", "srs"}
# Sources that need z-scoring onto the common scale.
SCALED_SOURCES = {"elo", "talent"}
ALL_SOURCES = sorted(NATIVE_POINT_SOURCES | SCALED_SOURCES)

# Fallback spread of team strength in points, used only when a season has too
# few SP+ rows to measure it (e.g. very early preseason).
DEFAULT_POINTS_SD = 10.5

_SUFFIX_FIXES = {
    "st": "state",
    "st.": "state",
    "u": "",
    "univ": "",
}


def normalize_team(name: str) -> str:
    """Canonical key for a team name so sources can be joined.

    Handles the ``Ohio St.`` / ``Ohio State`` and accent mismatches that show up
    when joining scraped spreadsheets against API data.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    # Apostrophes join (Hawai'i -> hawaii); other punctuation separates.
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    out = []
    for tok in tokens:
        mapped = _SUFFIX_FIXES.get(tok, tok)
        if mapped:
            out.append(mapped)
    return " ".join(out)


@dataclass
class TeamRating:
    team: str
    conference: str = ""
    values: Dict[str, float] = field(default_factory=dict)  # source -> points above avg
    meta: Dict[str, float] = field(default_factory=dict)  # extra context for the report

    def get(self, source: str) -> Optional[float]:
        return self.values.get(source)


class RatingBook:
    """All sources' ratings for one season, keyed by normalized team name."""

    def __init__(self, season: int) -> None:
        self.season = season
        self.teams: Dict[str, TeamRating] = {}
        self.display_names: Dict[str, str] = {}
        self.points_sd: float = DEFAULT_POINTS_SD
        # Raw values for sources that need z-scoring, held until finalize() so
        # the scaling never depends on which source happened to load first.
        self._pending: Dict[str, Dict[str, float]] = {}
        self._pending_names: Dict[str, str] = {}
        self._pending_confs: Dict[str, str] = {}

    # -- construction --------------------------------------------------------
    def _entry(self, team: str, conference: str = "") -> TeamRating:
        key = normalize_team(team)
        if key not in self.teams:
            self.teams[key] = TeamRating(team=team, conference=conference)
            self.display_names[key] = team
        elif conference and not self.teams[key].conference:
            self.teams[key].conference = conference
        return self.teams[key]

    def add(self, team: str, source: str, value: float, conference: str = "") -> None:
        self._entry(team, conference).values[source] = float(value)

    def add_meta(self, team: str, key: str, value: float, conference: str = "") -> None:
        self._entry(team, conference).meta[key] = float(value)

    def get(self, team: str) -> Optional[TeamRating]:
        return self.teams.get(normalize_team(team))

    def rating(self, team: str, source: str) -> Optional[float]:
        entry = self.get(team)
        return entry.get(source) if entry else None

    def sources_for(self, team: str) -> List[str]:
        entry = self.get(team)
        return sorted(entry.values) if entry else []

    def __len__(self) -> int:
        return len(self.teams)

    # -- loaders -------------------------------------------------------------
    def load_sp(self, rows: Iterable[dict]) -> int:
        count = 0
        for row in rows:
            team = pick(row, "team", "school")
            rating = pick_float(row, "rating")
            if not team or rating is None or normalize_team(team) == "nationalaverages":
                continue
            conf = pick(row, "conference", default="") or ""
            self.add(team, "sp_plus", rating, conf)
            for label, keys in (
                ("sp_offense", ("offense.rating", "offense")),
                ("sp_defense", ("defense.rating", "defense")),
                ("sp_special", ("specialTeams.rating", "special_teams.rating")),
            ):
                val = pick_float(row, *keys)
                if val is not None:
                    self.add_meta(team, label, val)
            count += 1
        self._recompute_scale()
        return count

    def load_fpi(self, rows: Iterable[dict]) -> int:
        count = 0
        for row in rows:
            team = pick(row, "team", "school")
            rating = pick_float(row, "fpi", "rating")
            if not team or rating is None:
                continue
            self.add(team, "fpi", rating, pick(row, "conference", default="") or "")
            for label, keys in (
                ("fpi_offense", ("efficiencies.offense", "offenseEfficiency")),
                ("fpi_defense", ("efficiencies.defense", "defenseEfficiency")),
                ("fpi_special", ("efficiencies.specialTeams",)),
            ):
                val = pick_float(row, *keys)
                if val is not None:
                    self.add_meta(team, label, val)
            count += 1
        return count

    def load_srs(self, rows: Iterable[dict]) -> int:
        count = 0
        for row in rows:
            team = pick(row, "team", "school")
            rating = pick_float(row, "rating")
            if not team or rating is None:
                continue
            self.add(team, "srs", rating, pick(row, "conference", default="") or "")
            count += 1
        return count

    def _stage(self, source: str, team: str, value: float, conference: str = "") -> None:
        key = normalize_team(team)
        self._pending.setdefault(source, {})[key] = float(value)
        self._pending_names[key] = team
        if conference:
            self._pending_confs[key] = conference

    def load_elo(self, rows: Iterable[dict]) -> int:
        """Stage the latest Elo per team; scaling happens in finalize()."""
        latest: Dict[str, tuple] = {}
        for row in rows:
            team = pick(row, "team", "school")
            elo = pick_float(row, "elo", "rating")
            if not team or elo is None:
                continue
            week = pick_float(row, "week", default=0.0) or 0.0
            key = normalize_team(team)
            if key not in latest or week >= latest[key][0]:
                latest[key] = (week, team, elo, pick(row, "conference", default="") or "")

        for _, team, elo, conf in latest.values():
            self._stage("elo", team, elo, conf)
            self.add_meta(team, "elo_raw", elo, conf)
        return len(latest)

    def load_talent(self, rows: Iterable[dict]) -> int:
        count = 0
        for row in rows:
            team = pick(row, "team", "school")
            talent = pick_float(row, "talent")
            if not team or talent is None:
                continue
            self._stage("talent", team, talent)
            self.add_meta(team, "talent_raw", talent)
            count += 1
        return count

    def finalize(self) -> None:
        """Scale staged sources onto the points scale set by SP+ dispersion.

        Safe to call more than once; staged raws are kept so re-running after a
        late-arriving SP+ load simply rescales.
        """
        self._recompute_scale()
        for source, raw in self._pending.items():
            for key, points in self._to_points(raw).items():
                team = self._pending_names.get(key, key)
                self.add(team, source, points, self._pending_confs.get(key, ""))

    def load_returning_production(self, rows: Iterable[dict]) -> int:
        count = 0
        for row in rows:
            team = pick(row, "team", "school")
            total = pick_float(row, "totalPPA", "total_ppa", "percentPPA", "percent_ppa")
            if not team or total is None:
                continue
            self.add_meta(team, "returning_ppa", total)
            count += 1
        return count

    # -- scaling -------------------------------------------------------------
    def _recompute_scale(self) -> None:
        sp_values = [t.values["sp_plus"] for t in self.teams.values() if "sp_plus" in t.values]
        if len(sp_values) >= 20:
            self.points_sd = _stdev(sp_values) or DEFAULT_POINTS_SD

    def _to_points(self, raw: Dict[str, float]) -> Dict[str, float]:
        """z-score a raw rating map, then stretch to the season's points SD."""
        if not raw:
            return {}
        values = list(raw.values())
        mean = sum(values) / len(values)
        sd = _stdev(values)
        if not sd:
            return {k: 0.0 for k in raw}
        return {k: (v - mean) / sd * self.points_sd for k, v in raw.items()}


def _stdev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def build_rating_book(client, season: int, week: Optional[int] = None) -> RatingBook:
    """Pull every rating source for a season into one book.

    A source that errors is logged and skipped rather than killing the run: a
    forecast off four of five sources still beats no forecast on Thursday
    morning. :func:`cfbmeta.model.project_game` renormalizes the weights over
    whichever sources actually showed up.
    """
    import logging

    log = logging.getLogger(__name__)
    book = RatingBook(season)

    loaders = [
        ("SP+", lambda: book.load_sp(client.sp_ratings(season))),
        ("FPI", lambda: book.load_fpi(client.fpi_ratings(season))),
        ("SRS", lambda: book.load_srs(client.srs_ratings(season))),
        ("Elo", lambda: book.load_elo(client.elo_ratings(season, week=week))),
        ("talent", lambda: book.load_talent(client.talent(season))),
        ("returning production", lambda: book.load_returning_production(
            client.returning_production(season))),
    ]

    for label, loader in loaders:
        try:
            n = loader()
            log.info("loaded %s for %d teams", label, n)
        except Exception as exc:  # noqa: BLE001 - degrade, don't die
            log.warning("could not load %s: %s", label, exc)

    book.finalize()
    return book
