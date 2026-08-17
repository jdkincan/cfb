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

# A source needs at least this many rated teams, and at least this much spread
# between them, to be worth blending. In August, CFBD will happily return an
# SRS or Elo row for every team with all of them identical, because no games
# have been played. Such a source predicts a tie in every game, and blending it
# drags every projection toward pick'em while looking perfectly healthy.
MIN_SOURCE_TEAMS = 10
MIN_SOURCE_DISPERSION = 0.5

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
class SourceStatus:
    """What actually arrived for one rating source, and whether it is usable.

    Recorded for every source so a preseason run can be audited at a glance:
    a missing FPI or a flat SRS should be visible, never silent.
    """

    source: str
    season: int
    teams: int = 0
    usable: bool = False
    dispersion: float = 0.0
    reason: str = ""
    origin: str = "CFBD"
    # Content hash and when these exact values were first seen, so a source
    # that is real but has stopped updating can be told apart from a fresh one.
    fingerprint: str = ""
    unchanged_since: str = ""
    unchanged_days: float = 0.0

    @property
    def stale(self) -> bool:
        """Real values that have not moved in over a week.

        A missing source and a flat source are both visible already. This is the
        third failure: a source serving genuine, well-dispersed, *old* numbers.
        In season every rating should move weekly, so standing still is a
        signal, not a comfort.
        """
        return self.usable and self.unchanged_days > 7.0

    def describe(self) -> str:
        label = SOURCE_DISPLAY.get(self.source, self.source)
        if not self.usable:
            return f"{label} {self.season}: unusable — {self.reason}"
        via = "" if self.origin == "CFBD" else f" via {self.origin}"
        age = ""
        if self.unchanged_days >= 1:
            age = f", unchanged {self.unchanged_days:.0f}d"
            if self.stale:
                age += " — STALE?"
        return (
            f"{label} {self.season}: {self.teams} teams"
            f" (spread {self.dispersion:.1f}){via}{age}"
        )


SOURCE_DISPLAY = {
    "sp_plus": "SP+",
    "fpi": "FPI",
    "elo": "Elo",
    "srs": "SRS",
    "talent": "Talent",
}


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
        # source -> SourceStatus, so a preseason run can be audited.
        self.provenance: Dict[str, SourceStatus] = {}

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

    def has_ratings(self, team: str) -> bool:
        """True only if the team carries at least one usable rating.

        A team can exist in the book on metadata alone — returning production
        loaded, ratings did not — and such a team must not be projected. It
        would otherwise come out as a 0.0 margin and read as a pick'em game.
        """
        entry = self.get(team)
        return bool(entry and entry.values)

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
        """Scale staged sources, then drop any source that carries no signal.

        Safe to call more than once; staged raws are kept so re-running after a
        late-arriving SP+ load simply rescales.
        """
        self._recompute_scale()
        for source, raw in self._pending.items():
            for key, points in self._to_points(raw).items():
                team = self._pending_names.get(key, key)
                self.add(team, source, points, self._pending_confs.get(key, ""))
        self._prune_degenerate_sources()

    def _prune_degenerate_sources(self) -> None:
        """Remove sources that are absent, too sparse, or completely flat.

        The flat case is the one that matters in August: a preseason SRS where
        every team sits at 0.0 is not a weak opinion, it is no opinion, and
        blending it would pull every projection toward a tie.
        """
        for source in ALL_SOURCES:
            values = [t.values[source] for t in self.teams.values() if source in t.values]
            if not values:
                self.provenance.setdefault(
                    source,
                    SourceStatus(source, self.season, 0, False, 0.0, "no data returned"),
                )
                continue

            dispersion = _stdev(values)
            if len(values) < MIN_SOURCE_TEAMS:
                reason = f"only {len(values)} teams rated"
            elif dispersion < MIN_SOURCE_DISPERSION:
                reason = (
                    f"all {len(values)} teams within {dispersion:.2f} pts "
                    "— no games played yet"
                )
            else:
                self.provenance[source] = SourceStatus(
                    source, self.season, len(values), True, round(dispersion, 2)
                )
                continue

            for entry in self.teams.values():
                entry.values.pop(source, None)
            self.provenance[source] = SourceStatus(
                source, self.season, len(values), False, round(dispersion, 2), reason
            )

    def usable_sources(self) -> List[str]:
        return sorted(s for s, p in self.provenance.items() if p.usable)

    def provenance_lines(self) -> List[str]:
        return [self.provenance[s].describe() for s in sorted(self.provenance)]

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


def _try_espn_fpi(book: "RatingBook", season: int) -> None:
    """Fill FPI from ESPN when CFBD has none, for teams already in the book.

    Only known teams are filled. ESPN's naming differs in places ("Miami (FL)"),
    and creating book entries from unmatched names would invent teams that no
    game ever refers to.
    """
    import logging

    log = logging.getLogger(__name__)
    previous = book.provenance.get("fpi")
    try:
        from .sources.espn import fetch_fpi

        rows = fetch_fpi(season)
    except Exception as exc:  # noqa: BLE001
        log.warning("ESPN FPI fallback failed: %s", exc)
        return
    if not rows:
        return

    matched = 0
    for row in rows:
        team = row.get("team")
        value = row.get("fpi")
        if team is None or value is None:
            continue
        # Only teams CFBD already gave us; never introduce new ones.
        if normalize_team(team) in book.teams:
            book.add(book.teams[normalize_team(team)].team, "fpi", float(value))
            matched += 1

    if not matched:
        log.warning("ESPN FPI returned %d teams, none matched the book", len(rows))
        return

    log.info("ESPN FPI fallback filled %d of %d teams", matched, len(book))
    book._prune_degenerate_sources()
    status = book.provenance.get("fpi")
    if status is not None and status.usable:
        status.origin = "ESPN"
    elif status is not None and previous is not None:
        # Fallback didn't rescue it; keep whichever reason is more informative.
        status.reason = f"{status.reason} (ESPN fallback matched {matched} teams)"


def _try_recruiting_talent(book: "RatingBook", client, season: int) -> None:
    """Rebuild talent from the current roster when /talent is empty.

    Built from who is actually on the team, not from signing classes, so
    transfers in and out are handled correctly. Reproduces CFBD's own composite
    at r = 0.984. See sources/recruiting.py.
    """
    import logging

    log = logging.getLogger(__name__)
    try:
        from .sources.recruiting import build_roster_talent, talent_rows

        profiles = build_roster_talent(client, season)
        rows = talent_rows(profiles)
    except Exception as exc:  # noqa: BLE001
        log.warning("recruiting talent fallback failed: %s", exc)
        return
    if not rows:
        return

    book.load_talent(rows)
    for key, prof in profiles.items():
        if prof.blue_chip_ratio is not None and key in book.teams:
            book.teams[key].meta["blue_chip_ratio"] = prof.blue_chip_ratio
    book.finalize()
    status = book.provenance.get("talent")
    if status is not None and status.usable:
        status.origin = "roster x recruiting"


def _record_freshness(book: "RatingBook") -> None:
    """Note whether each source's values have moved since the last run."""
    import logging

    log = logging.getLogger(__name__)
    try:
        from .freshness import fingerprint, load_state, record, save_state
    except Exception:  # noqa: BLE001
        return

    state = load_state()
    for source, status in book.provenance.items():
        if not status.usable:
            continue
        values = {
            team: entry.values[source]
            for team, entry in book.teams.items()
            if source in entry.values
        }
        if not values:
            continue
        digest = fingerprint(values)
        days = record(f"{book.season}:{source}", digest, state)
        status.fingerprint = digest
        status.unchanged_days = round(days, 1)
        if status.stale:
            log.warning(
                "%s has served identical values for %.0f days — check upstream",
                source, days,
            )
    save_state(state)


def build_rating_book(
    client,
    season: int,
    week: Optional[int] = None,
    espn_fpi_fallback: bool = True,
    recruiting_talent_fallback: bool = True,
) -> RatingBook:
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
        ("SP+", "sp_plus", lambda: book.load_sp(client.sp_ratings(season))),
        ("FPI", "fpi", lambda: book.load_fpi(client.fpi_ratings(season))),
        ("SRS", "srs", lambda: book.load_srs(client.srs_ratings(season))),
        ("Elo", "elo", lambda: book.load_elo(client.elo_ratings(season, week=week))),
        ("talent", "talent", lambda: book.load_talent(client.talent(season))),
        ("returning production", None, lambda: book.load_returning_production(
            client.returning_production(season))),
    ]

    for label, source, loader in loaders:
        try:
            n = loader()
            log.info("loaded %s for %d teams (season %d)", label, n, season)
        except Exception as exc:  # noqa: BLE001 - degrade, don't die
            log.warning("could not load %s for %d: %s", label, season, exc)
            if source:
                book.provenance[source] = SourceStatus(
                    source, season, 0, False, 0.0, f"request failed: {exc}"
                )

    book.finalize()

    # CFBD mirrors ESPN's FPI on its own schedule and can lag at the start of a
    # season. If it came back unusable, try ESPN directly before giving up on
    # the source entirely.
    if espn_fpi_fallback and not book.provenance.get("fpi", SourceStatus("fpi", season)).usable:
        _try_espn_fpi(book, season)

    if (
        recruiting_talent_fallback
        and not book.provenance.get("talent", SourceStatus("talent", season)).usable
    ):
        _try_recruiting_talent(book, client, season)

    _record_freshness(book)
    for line in book.provenance_lines():
        log.info("  %s", line)
    if not book.usable_sources():
        log.error("no usable rating source for season %d", season)
    return book
