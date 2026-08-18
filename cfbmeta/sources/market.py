"""Sportsbook lines: consensus, best available price, and how far it has moved.

Three different numbers matter and they are not interchangeable.

**Consensus** (median across books) is the right benchmark for *measuring*
edge. It is the market's opinion, robust to one stale or mis-keyed quote.

**Best available** is the right number for *betting*. You do not bet the
median; you bet a specific number at a specific book, and taking +7 where the
consensus is +6.5 is a half point of free equity. The catch is that the best
number is sometimes best because it is stale or wrong, so the two are reported
side by side rather than one replacing the other.

**Movement** is the opening number against the current one. CFBD exposes
``spreadOpen``, so this needs no extra capture: a line that has moved toward
our side since open means the market is agreeing with us, and a line moving
away means it is disagreeing — which historically is the more informative
direction.

A note on availability: CFBD's free tier carries DraftKings, ESPN Bet and
Bovada. FanDuel is not among them, so a FanDuel-first preference cannot be
honoured; the preference order below falls back to DraftKings as the closest
widely-available substitute.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from ..ratings import normalize_team
from .cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Preference order when two books offer the same number. FanDuel is listed
# first so it is used the moment CFBD ever carries it; today it never matches.
PREFERRED_PROVIDERS = ("fanduel", "draftkings", "espn bet", "bovada", "consensus", "caesars")


def _provider_rank(name: str) -> int:
    lowered = (name or "").lower()
    for i, pref in enumerate(PREFERRED_PROVIDERS):
        if pref in lowered:
            return i
    return len(PREFERRED_PROVIDERS)


def best_spread(quotes: List[dict], side: str) -> Optional[Dict[str, Any]]:
    """The most favourable number available for one side, and where.

    ``side`` is "home" or "away". Spreads are quoted from the home side, so the
    home bettor wants the largest spread (most points, or fewest laid) and the
    away bettor wants the smallest.
    """
    priced = [
        (pick_float(q, "spread"), pick(q, "provider", "providerName", default="") or "")
        for q in quotes
        if pick_float(q, "spread") is not None
    ]
    if not priced:
        return None
    # Sort by the number first, then book preference to break exact ties.
    if side == "home":
        priced.sort(key=lambda p: (-p[0], _provider_rank(p[1])))
    else:
        priced.sort(key=lambda p: (p[0], _provider_rank(p[1])))
    spread, provider = priced[0]
    return {"spread": round(spread, 2), "provider": provider}


def consensus_line(row: dict) -> Optional[Dict[str, Any]]:
    """Median spread/total across books, plus best price per side and movement."""
    quotes = pick(row, "lines", default=[]) or []
    spreads: List[float] = []
    totals: List[float] = []
    opens: List[float] = []
    providers: List[str] = []

    for quote in quotes:
        spread = pick_float(quote, "spread")
        total = pick_float(quote, "overUnder", "over_under", "total")
        opening = pick_float(quote, "spreadOpen", "spread_open")
        provider = pick(quote, "provider", "providerName", default="") or ""
        if spread is not None:
            spreads.append(spread)
            providers.append(provider)
        if total is not None:
            totals.append(total)
        if opening is not None:
            opens.append(opening)

    if not spreads and not totals:
        return None

    providers.sort(key=_provider_rank)
    consensus = round(statistics.median(spreads), 2) if spreads else None
    opening = round(statistics.median(opens), 2) if opens else None

    return {
        "spread": consensus,
        "total": round(statistics.median(totals), 2) if totals else None,
        "provider": f"median of {len(spreads)}" if len(spreads) > 1 else (providers[0] if providers else ""),
        "book_count": len(spreads),
        # Which books, not just how many. A second odds source has to be able
        # to tell "DraftKings again" from "a book CFBD does not carry", or the
        # book count inflates every time the feeds overlap.
        "books": sorted({p for p in providers if p}),
        "spread_range": (round(min(spreads), 2), round(max(spreads), 2)) if spreads else None,
        "spread_open": opening,
        # Positive means the number has moved toward the home side since open.
        "movement": round(-(consensus - opening), 2)
        if consensus is not None and opening is not None else None,
        "best_home": best_spread(quotes, "home"),
        "best_away": best_spread(quotes, "away"),
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
