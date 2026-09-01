"""Weekly snapshots of everything a run saw.

CFBD is a point-in-time API: ``/ratings/sp?year=2026`` returns whatever SP+ is
*today*, and last week's values are gone. There is no ``as_of`` parameter. That
has a consequence bigger than inconvenience — it makes the model impossible to
validate honestly.

The backtest can only rate a past game with either that season's *final*
ratings (lookahead — they already know the result) or the *previous* season's
final ratings (fair, but far weaker than what a live Thursday run actually
has). Neither is the real thing. The question that matters — "given the ratings
that existed on the Thursday before kickoff, did this model beat the closing
line?" — cannot be answered from the API at all.

It can only be answered from data we keep ourselves. So every run writes down
what it saw: the ratings, the market, the projections, and enough game context
to grade them later. After a season of Thursdays, ``backtest --basis snapshot``
becomes possible, and that is the first result worth trusting.

Layout, one directory per season and week::

    archive/2026/week-01/ratings.csv       team, source, rating
                         lines.csv         game, book consensus, range
                         projections.csv   the full readout, one row per game
                         meta.json         source provenance, config, timestamps

CSV so it loads straight into pandas or a spreadsheet; JSON for the nested
provenance that doesn't flatten well.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent / "archive"


def week_dir(season: int, week: int, root: Optional[Path] = None) -> Path:
    return (root or ARCHIVE_ROOT) / str(season) / f"week-{int(week):02d}"


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def ratings_rows(book) -> List[Dict[str, Any]]:
    """One row per team per source, as the values stood at this moment."""
    rows = []
    for key, entry in book.teams.items():
        for source, value in sorted(entry.values.items()):
            rows.append({
                "team": entry.team,
                "team_key": key,
                "conference": entry.conference,
                "source": source,
                "rating": round(value, 4),
            })
        for name, value in sorted(entry.meta.items()):
            rows.append({
                "team": entry.team,
                "team_key": key,
                "conference": entry.conference,
                "source": f"meta:{name}",
                "rating": round(value, 4),
            })
    return rows


def projection_rows(projections: Iterable) -> List[Dict[str, Any]]:
    """The readout itself, flattened — one row per game, gradeable later."""
    rows = []
    for proj in projections:
        bet = proj.spread_bet
        row = {
            "game_id": proj.game_id,
            "season": proj.season,
            "week": proj.week,
            "kickoff": proj.kickoff.isoformat() if proj.kickoff else "",
            "home_team": proj.home_team,
            "away_team": proj.away_team,
            "neutral_site": proj.neutral_site,
            "venue": proj.venue,
            "neutral_margin": proj.neutral_margin,
            "hfa": (proj.hfa or {}).get("total", 0.0),
            "hfa_elevation": (proj.hfa or {}).get("elevation", 0.0),
            "coaching": (proj.coaching or {}).get("total", 0.0),
            "situational": (proj.situational or {}).get("total", 0.0),
            "adjustment_total": proj.adjustment_total,
            "projected_margin": proj.projected_margin,
            "projected_total": proj.projected_total,
            "market_spread": proj.market_spread,
            "market_total": proj.market_total,
            "edge": proj.edge,
            "home_win_probability": proj.home_win_probability,
            "bet_side": bet.side if bet else "none",
            "bet_line": bet.line if bet else None,
            "cover_probability": bet.cover_probability if bet else None,
            "expected_value": bet.expected_value if bet else None,
            "units": bet.units if bet else 0.0,
            # Graded after the fact by `archive grade`.
            "home_points": "",
            "away_points": "",
            "actual_margin": "",
            "bet_result": "",
        }
        for comp in proj.components:
            row[f"component_{comp.source}"] = comp.margin
            row[f"weight_{comp.source}"] = round(comp.weight, 4)
        rows.append(row)
    return rows


def market_rows(markets: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Consensus lines, keyed by game id only (the team-pair keys are aliases)."""
    rows = []
    for key, line in (markets or {}).items():
        if isinstance(key, tuple):
            continue
        spread_range = line.get("spread_range") or (None, None)
        rows.append({
            "game_id": key,
            "spread": line.get("spread"),
            "total": line.get("total"),
            "provider": line.get("provider", ""),
            "book_count": line.get("book_count", 0),
            "spread_min": spread_range[0],
            "spread_max": spread_range[1],
        })
    return rows


def _keep_first(path: Path, rows: List[dict], columns: List[str],
                stamp: str) -> int:
    """Write ``path`` once; later captures go to a timestamped sibling.

    The first capture of a week is the decision snapshot — the numbers that
    existed when the forecast was made — and it is the only one that can ever
    be graded honestly. Overwriting it with a later re-run silently replaces
    "what we knew on Thursday" with "what we knew after the lines moved",
    which is the one thing this whole directory exists to prevent.
    """
    if not path.exists():
        return _write_csv(path, rows, columns)
    later = path.with_name(f"{path.stem}-{stamp}{path.suffix}")
    return _write_csv(later, rows, columns)


def _append_history(path: Path, rows: List[dict], columns: List[str],
                    stamp: str) -> int:
    """Append this capture to a running history, one row per game per capture.

    Line movement between the forecast and kickoff is the raw material for
    closing-line value, and it only exists if every capture is kept.
    """
    header = ["captured_at"] + columns
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"captured_at": stamp, **row})
    return len(rows)


def snapshot(
    season: int,
    week: int,
    book=None,
    projections: Optional[Iterable] = None,
    markets: Optional[Dict] = None,
    config=None,
    root: Optional[Path] = None,
    now: Optional[dt.datetime] = None,
) -> Path:
    """Persist everything this run saw. Returns the directory written."""
    now = now or dt.datetime.now(dt.timezone.utc)
    directory = week_dir(season, week, root)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    counts = {}
    if book is not None:
        counts["ratings"] = _keep_first(
            directory / "ratings.csv",
            ratings_rows(book),
            ["team", "team_key", "conference", "source", "rating"],
            stamp,
        )

    projections = list(projections or [])
    if projections:
        rows = projection_rows(projections)
        columns: List[str] = []
        for row in rows:  # union of keys, stable order
            for key in row:
                if key not in columns:
                    columns.append(key)
        counts["projections"] = _keep_first(
            directory / "projections.csv", rows, columns, stamp)

    if markets:
        line_columns = ["game_id", "spread", "total", "provider", "book_count",
                        "spread_min", "spread_max"]
        line_rows = market_rows(markets)
        counts["lines"] = _keep_first(
            directory / "lines.csv", line_rows, line_columns, stamp)
        # Every capture, always — this is the movement series.
        counts["line_history"] = _append_history(
            directory / "lines-history.csv", line_rows, line_columns, stamp)

    previous = {}
    meta_path = directory / "meta.json"
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            previous = {}
    captures = list(previous.get("captures") or [])
    if previous.get("captured_at") and not captures:
        captures.append(previous["captured_at"])
    captures.append(now.isoformat())

    meta = {
        # The first capture is the one a grade is scored against; the rest are
        # kept so line movement can be reconstructed.
        "captured_at": previous.get("captured_at") or now.isoformat(),
        "captures": captures,
        "season": season,
        "week": week,
        "counts": counts,
        "sources": (
            {k: asdict(v) for k, v in book.provenance.items()} if book is not None else {}
        ),
        "usable_sources": book.usable_sources() if book is not None else [],
    }
    if config is not None:
        meta["config"] = {
            "weights": asdict(config.weights),
            "sigma_margin": config.sigma_margin,
            "edge_shrink": config.edge_shrink,
            "league_hfa": config.league_hfa,
            "min_edge_points": config.min_edge_points,
        }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    log.info("archived %s: %s", directory, counts)
    return directory


def list_snapshots(season: Optional[int] = None, root: Optional[Path] = None) -> List[Path]:
    base = root or ARCHIVE_ROOT
    if not base.exists():
        return []
    seasons = [base / str(season)] if season else sorted(p for p in base.iterdir() if p.is_dir())
    out: List[Path] = []
    for directory in seasons:
        if directory.is_dir():
            out.extend(sorted(p for p in directory.iterdir() if p.is_dir()))
    return out


def grade(season: int, week: int, games: Iterable[dict], root: Optional[Path] = None) -> int:
    """Fill in final scores and bet results on a stored week.

    Run after the games are played; this is what converts a snapshot from a
    record of what we thought into a record of whether we were right.
    """
    from .sources.cfbd import pick, pick_float

    directory = week_dir(season, week, root)
    path = directory / "projections.csv"
    if not path.exists():
        log.warning("no snapshot to grade at %s", path)
        return 0

    results = {}
    for game in games:
        gid = pick(game, "id", "gameId")
        hp = pick_float(game, "homePoints", "home_points")
        ap = pick_float(game, "awayPoints", "away_points")
        if gid is not None and hp is not None and ap is not None:
            results[str(gid)] = (hp, ap)

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
        columns = rows[0].keys() if rows else []

    graded = 0
    for row in rows:
        outcome = results.get(str(row.get("game_id")))
        if not outcome:
            continue
        home_points, away_points = outcome
        margin = home_points - away_points
        row["home_points"] = home_points
        row["away_points"] = away_points
        row["actual_margin"] = margin
        row["bet_result"] = _grade_bet(row, margin)
        graded += 1

    if graded:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
    log.info("graded %d games in %s", graded, directory)
    return graded


def _grade_bet(row: Dict[str, Any], margin: float) -> str:
    """win / loss / push for the side the model took, or '' if it passed."""
    side = row.get("bet_side") or "none"
    if side == "none" or not row.get("market_spread"):
        return ""
    try:
        market_margin = -float(row["market_spread"])
    except (TypeError, ValueError):
        return ""
    if margin == market_margin:
        return "push"
    home_covered = margin > market_margin
    if side == "home":
        return "win" if home_covered else "loss"
    if side == "away":
        return "loss" if home_covered else "win"
    return ""


def write_efficiency_series(season: int, series: Dict[int, Any],
                            root: Optional[Path] = None) -> Path:
    """Persist the weekly efficiency ratings as one long CSV.

    Long rather than wide (one row per team per week) because the number of
    weeks is not known ahead of time and a wide file has to be rewritten
    whenever the season grows. This loads straight into pandas for plotting a
    team's arc across the year.
    """
    root = Path(root) if root else ARCHIVE_ROOT
    directory = root / str(season)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "efficiency-weekly.csv"

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["season", "through_week", "team", "rating", "rank", "offense",
             "defense", "points_per_unit", "observations"]
        )
        for week in sorted(series):
            ratings = series[week]
            ranked = ratings.ranked()
            for rank, (team, value) in enumerate(ranked, 1):
                writer.writerow([
                    season, week - 1, team, round(value, 3), rank,
                    round(ratings.offense.get(team, 0.0), 4),
                    round(ratings.defense.get(team, 0.0), 4),
                    round(ratings.points_per_unit, 2),
                    ratings.observations,
                ])
    log.info("wrote efficiency series to %s", path)
    return path
