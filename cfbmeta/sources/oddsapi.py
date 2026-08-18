"""Odds from the-odds-api.com: more books, and a sharp reference price.

CFBD's line feed carries three retail books — DraftKings, ESPN Bet, Bovada.
That is enough to compute a median and not much else, and it has two problems
this module exists to fix.

**Line shopping.** You do not bet a median, you bet a number at a book. Across
ten books the best available number is routinely a half point better than the
median, and on a key number a half point is worth more than most of what the
model claims to know. This is a mechanical edge that requires no forecasting
skill at all, which makes it the most reliable one available.

**A sharp benchmark.** More important and less obvious. Measuring "edge"
against a median of soft retail books conflates two different things: the model
being right, and those books being slow. Pinnacle takes large limits and moves
on sharp money, so its number is the closest thing to a fair price. Benchmark
against Pinnacle and the edge you measure is real edge; benchmark against a
median of retail books and you cannot tell the difference.

So when this source is configured, the consensus used for grading becomes the
sharp book's number where one is available, while stakes still size against the
best number found anywhere.

Cost, as of this writing: free tier 500 credits/month with most bookmakers,
$30/month for 20,000 credits and all bookmakers. A weekly college football
pull costs a handful of credits, so the free tier covers the scheduled job;
the paid tier is about *which* books, not how many calls.

Set ODDS_API_KEY to enable. Absent that, everything below no-ops and CFBD's
lines are used unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests

from ..ratings import normalize_team

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_ncaaf"
# Books that move on sharp money, best first. The first one present in a
# payload becomes the benchmark price for measuring edge.
SHARP_BOOKS = ("pinnacle", "betonlineag", "lowvig", "circasports")


@dataclass
class BookQuote:
    book: str
    spread: float          # from the home side, sportsbook convention
    price: int = -110
    total: Optional[float] = None


@dataclass
class GameOdds:
    home_team: str
    away_team: str
    quotes: List[BookQuote] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (normalize_team(self.home_team), normalize_team(self.away_team))

    def sharp_spread(self) -> Optional[BookQuote]:
        """The sharpest book's number, if one of them quoted this game."""
        for name in SHARP_BOOKS:
            for quote in self.quotes:
                if name in quote.book.lower().replace(" ", ""):
                    return quote
        return None


def _book_id(name: str) -> str:
    """Normalise a book name so two feeds' spellings of it collide.

    CFBD says "ESPN Bet", the odds feed says "espnbet". Both have to reduce to
    the same token or every overlapping book is counted twice.
    """
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _match_team(name: str, known: Iterable[str]) -> Optional[str]:
    """Map "Alabama Crimson Tide" onto CFBD's "Alabama".

    The odds feed carries full names with mascots; CFBD does not. Exact match
    first, then the longest known school name that prefixes the odds name,
    which resolves "Miami (FL) Hurricanes" and "Ohio State Buckeyes" alike
    without a hand-maintained alias table.
    """
    known = set(known)
    key = normalize_team(name)
    if key in known:
        return key
    prefixed = [k for k in known if key.startswith(k + " ")]
    if prefixed:
        return max(prefixed, key=len)
    # Fall back to the longest known name contained in the odds name.
    contained = [k for k in known if f" {k} " in f" {key} "]
    return max(contained, key=len) if contained else None


def fetch_odds(
    api_key: Optional[str] = None,
    regions: str = "us,us2",
    timeout: int = 30,
    session=None,
) -> List[GameOdds]:
    """Current NCAAF spreads across every book the plan covers."""
    api_key = api_key or os.getenv("ODDS_API_KEY") or ""
    if not api_key:
        return []

    http = session or requests
    try:
        response = http.get(
            f"{BASE_URL}/sports/{SPORT_KEY}/odds",
            params={
                "apiKey": api_key,
                "regions": regions,
                "markets": "spreads,totals",
                "oddsFormat": "american",
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            log.warning(
                "odds api returned %s: %s", response.status_code, response.text[:160]
            )
            return []
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("odds api fetch failed: %s", exc)
        return []

    remaining = None
    try:
        remaining = response.headers.get("x-requests-remaining")
    except Exception:  # noqa: BLE001
        pass
    if remaining is not None:
        log.info("odds api: %s credits remaining this period", remaining)

    return parse_odds(payload)


def parse_odds(payload: Any) -> List[GameOdds]:
    """Turn the odds-api payload into per-game book quotes."""
    games: List[GameOdds] = []
    for event in payload or []:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            continue
        game = GameOdds(home_team=home, away_team=away)

        for book in event.get("bookmakers") or []:
            title = book.get("key") or book.get("title") or ""
            spread = price = total = None
            for market in book.get("markets") or []:
                if market.get("key") == "spreads":
                    for outcome in market.get("outcomes") or []:
                        if outcome.get("name") == home:
                            spread = outcome.get("point")
                            price = outcome.get("price", -110)
                elif market.get("key") == "totals":
                    outcomes = market.get("outcomes") or []
                    if outcomes:
                        total = outcomes[0].get("point")
            if spread is not None:
                game.quotes.append(BookQuote(
                    book=title, spread=float(spread),
                    price=int(price) if price is not None else -110,
                    total=float(total) if total is not None else None,
                ))
        if game.quotes:
            games.append(game)
    return games


def merge_into_markets(
    markets: Dict[Any, Dict[str, Any]],
    odds: Iterable[GameOdds],
    prefer_sharp_benchmark: bool = True,
) -> int:
    """Fold odds-api books into an existing CFBD market map.

    Widens the book set used for best-price selection, and — when a sharp book
    quoted the game — replaces the grading benchmark with its number. Entries
    are mutated in place, so both the game-id and team-pair keys pointing at
    the same dict see the update. Returns how many games were enriched.
    """
    by_pair: Dict[tuple, Dict[str, Any]] = {
        k: v for k, v in markets.items() if isinstance(k, tuple) and len(k) == 2
    }
    if not by_pair:
        return 0

    home_names = {k[0] for k in by_pair}
    away_names = {k[1] for k in by_pair}
    all_names = home_names | away_names

    enriched = 0
    for game in odds:
        home = _match_team(game.home_team, all_names) or normalize_team(game.home_team)
        away = _match_team(game.away_team, all_names) or normalize_team(game.away_team)

        entry = by_pair.get((home, away))
        flip = False
        if entry is None:
            # Neutral-site games get their host assigned differently by each
            # feed. Same matchup, opposite orientation — usable, but every
            # spread has to be negated to stay quoted from CFBD's home side.
            entry = by_pair.get((away, home))
            flip = entry is not None
        if entry is None:
            continue

        sign = -1.0 if flip else 1.0
        spreads = [sign * q.spread for q in game.quotes]
        # Best number per side across the wider book set. Spreads are quoted
        # from the home side, so the home bettor wants the largest number.
        best_home_spread = max(spreads)
        best_away_spread = min(spreads)
        best_home_book = game.quotes[spreads.index(best_home_spread)].book
        best_away_book = game.quotes[spreads.index(best_away_spread)].book

        current_home = (entry.get("best_home") or {}).get("spread")
        current_away = (entry.get("best_away") or {}).get("spread")
        if current_home is None or best_home_spread > current_home:
            entry["best_home"] = {"spread": round(best_home_spread, 2),
                                  "provider": best_home_book}
        if current_away is None or best_away_spread < current_away:
            entry["best_away"] = {"spread": round(best_away_spread, 2),
                                  "provider": best_away_book}

        # Union the books rather than summing counts: the two feeds overlap on
        # DraftKings and Bovada, and adding them would report ten books where
        # there are seven.
        books = {_book_id(b): b for b in entry.get("books") or []}
        for quote in game.quotes:
            books.setdefault(_book_id(quote.book), quote.book)
        entry["books"] = sorted(books.values())
        entry["book_count"] = len(books)

        low, high = min(spreads), max(spreads)
        existing_range = entry.get("spread_range")
        if existing_range:
            low = min(low, existing_range[0])
            high = max(high, existing_range[1])
        entry["spread_range"] = (round(low, 2), round(high, 2))

        sharp = game.sharp_spread()
        if sharp is not None and prefer_sharp_benchmark:
            # Grade against the sharpest available price, not a soft median.
            # A median of slow retail books cannot distinguish "the model is
            # right" from "those books have not moved yet".
            entry["spread"] = round(sign * sharp.spread, 2)
            entry["provider"] = f"{sharp.book} (sharp)"
            entry["sharp_book"] = sharp.book
        enriched += 1

    log.info("odds api enriched %d game(s)", enriched)
    return enriched


def enrich(
    markets: Dict[Any, Dict[str, Any]],
    api_key: Optional[str] = None,
    regions: str = "us,us2",
    prefer_sharp_benchmark: bool = True,
    timeout: int = 30,
    session=None,
) -> int:
    """Fetch and merge in one call. A no-op without ODDS_API_KEY."""
    if not (api_key or os.getenv("ODDS_API_KEY")):
        return 0
    odds = fetch_odds(api_key=api_key, regions=regions, timeout=timeout, session=session)
    if not odds:
        return 0
    return merge_into_markets(
        markets, odds, prefer_sharp_benchmark=prefer_sharp_benchmark
    )
