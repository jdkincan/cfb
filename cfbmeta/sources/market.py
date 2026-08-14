"""Consolidate sportsbook lines into one market number per game.

CFBD returns a quote per book. Books disagree by a half point or so and the
occasional stale or mis-keyed quote is much further off, so the consensus here
is the *median* spread rather than the mean — one bad quote then moves nothing.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from ..ratings import normalize_team
from .cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Books listed first win ties when reporting which provider we quoted.
PREFERRED_PROVIDERS = ("consensus", "draftkings", "bovada", "espn bet", "fanduel", "caesars")


def _provider_rank(name: str) -> int:
    lowered = (name or "").lower()
    for i, pref in enumerate(PREFERRED_PROVIDERS):
        if pref in lowered:
            return i
    return len(PREFERRED_PROVIDERS)


def consensus_line(row: dict) -> Optional[Dict[str, Any]]:
    """Median spread/total across the books quoting one game."""
    quotes = pick(row, "lines", default=[]) or []
    spreads: List[float] = []
    totals: List[float] = []
    providers: List[str] = []

    for quote in quotes:
        spread = pick_float(quote, "spread")
        total = pick_float(quote, "overUnder", "over_under", "total")
        provider = pick(quote, "provider", "providerName", default="") or ""
        if spread is not None:
            spreads.append(spread)
            providers.append(provider)
        if total is not None:
            totals.append(total)

    if not spreads and not totals:
        return None

    providers.sort(key=_provider_rank)
    return {
        "spread": round(statistics.median(spreads), 2) if spreads else None,
        "total": round(statistics.median(totals), 2) if totals else None,
        "provider": f"median of {len(spreads)}" if len(spreads) > 1 else (providers[0] if providers else ""),
        "book_count": len(spreads),
        "spread_range": (round(min(spreads), 2), round(max(spreads), 2)) if spreads else None,
    }


def build_market_map(rows: List[dict]) -> Dict[Any, Dict[str, Any]]:
    """Map game id (and a team-pair fallback key) to the consensus line."""
    markets: Dict[Any, Dict[str, Any]] = {}
    for row in rows or []:
        line = consensus_line(row)
        if not line:
            continue
        gid = pick(row, "id", "gameId", "game_id")
        if gid is not None:
            markets[gid] = line
        home = pick(row, "homeTeam", "home_team")
        away = pick(row, "awayTeam", "away_team")
        if home and away:
            markets[(normalize_team(home), normalize_team(away))] = line
    return markets


def load_markets(client, season: int, week: int, season_type: str = "regular") -> Dict[Any, Dict[str, Any]]:
    try:
        rows = client.lines(season, week=week, season_type=season_type)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load betting lines: %s", exc)
        return {}
    markets = build_market_map(rows)
    log.info("loaded market lines for %d games", len({k for k in markets if not isinstance(k, tuple)}))
    return markets
