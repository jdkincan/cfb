"""Scrape posted season win totals across books.

Books block direct API access (DraftKings, FanDuel and BetMGM all answer 403
to anything that is not a browser), so the numbers come from an aggregator that
publishes them: VegasInsider's win-totals table carries all 138 FBS teams
across BetMGM, DraftKings, Caesars, FanDuel and RiversCasino.

Two things about this source shape everything below.

**Books disagree on the number, not just the juice.** Thirty of 139 teams are
posted a full win apart somewhere — Georgia at 9.5 and 10.5, Texas Tech at 10.5
and 11.5. That is worth far more than any price difference, and it is why the
best over and the best under for one team are often at *different totals*: an
over bettor wants the lowest number available, an under bettor the highest.

**Only over prices are published.** The table has no under juice at all, so an
under price has to be estimated from the over price and an assumed hold. That
estimate is flagged on every line it touches, and manual entry in
win-totals.yml always wins — if you are betting an under, read the real price
off your book rather than trusting a number this module inferred.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import requests

from ..probability import implied_probability
from ..ratings import normalize_team

log = logging.getLogger(__name__)

SOURCE_URL = "https://www.vegasinsider.com/college-football/odds/win-totals/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
# Typical two-way hold on a season win total. Higher than a side, because these
# are low-limit futures. Used only to estimate an unpublished under price.
DEFAULT_HOLD = 0.045

# Names the aggregator and CFBD genuinely spell differently. Kept small and
# explicit: every entry here is a case where fuzzy matching was wrong, not a
# case where it was merely unsure.
ALIASES = {
    "louisiana monroe": "ul monroe",
    "ul lafayette": "louisiana",
    "louisiana lafayette": "louisiana",
    "appalachian state": "app state",
    "umass": "massachusetts",
    "miami fl": "miami",
    "miami florida": "miami",
    "connecticut": "uconn",
    "southern methodist": "smu",
    "central florida": "ucf",
}

_ROW = re.compile(r'data-name="([^"]+)"')
_VALUE = re.compile(r'class="data-value">\s*([^<]+?)\s*</span>')
_HEADER_BOOK = re.compile(r"<thead.*?</thead>", re.S)


@dataclass
class BookTotal:
    book: str
    total: float
    over_price: int
    under_price: Optional[int] = None
    under_estimated: bool = False


@dataclass
class ScrapedTotal:
    team: str
    quotes: List[BookTotal] = field(default_factory=list)

    @property
    def key(self) -> str:
        return normalize_team(self.team)

    def best_over(self) -> Optional[BookTotal]:
        """Lowest number available, then the best price at that number."""
        if not self.quotes:
            return None
        return min(self.quotes, key=lambda q: (q.total, -q.over_price))

    def best_under(self) -> Optional[BookTotal]:
        """Highest number available, then the best price at that number."""
        if not self.quotes:
            return None
        return max(
            self.quotes,
            key=lambda q: (q.total, q.under_price if q.under_price is not None else -999),
        )

    @property
    def totals_disagree(self) -> bool:
        return len({q.total for q in self.quotes}) > 1


def probability_to_american(p: float) -> int:
    """Inverse of implied_probability, rounded to a real price."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    # Even money is quoted +100, not -100; both describe the same probability
    # but only one is a price anyone writes down.
    if p > 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def estimate_under_price(over_price: int, hold: float = DEFAULT_HOLD) -> int:
    """Infer the under price from the over price and an assumed hold.

    A two-way market's implied probabilities sum to 1 + hold. Given one side,
    that pins the other. It is an assumption, not a quote, and every line built
    this way is marked estimated.
    """
    implied_under = (1.0 + hold) - implied_probability(over_price)
    return probability_to_american(implied_under)


def book_names(html: str) -> List[str]:
    """Column order from the table header, so quotes can be attributed."""
    head = _HEADER_BOOK.search(html or "")
    if not head:
        return []
    cells = re.findall(r"<th[^>]*>(.*?)</th>", head.group(), re.S)
    names = []
    for cell in cells:
        text = re.sub(r"<[^>]+>", " ", cell)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text.lower() not in ("time", "team", "open"):
            names.append(text)
    return names


def parse(html: str, hold: float = DEFAULT_HOLD) -> List[ScrapedTotal]:
    """Pull one row per team out of the aggregator's table."""
    books = book_names(html)
    out: List[ScrapedTotal] = []

    for chunk in re.split(r"<tr\b", html or ""):
        match = _ROW.search(chunk)
        if not match:
            continue
        values = _VALUE.findall(chunk)
        entry = ScrapedTotal(team=match.group(1))
        # Values alternate line, price, line, price ... one pair per book.
        for i in range(0, len(values) - 1, 2):
            line, price = values[i], values[i + 1]
            m_line = re.fullmatch(r"[ou]?(\d{1,2}(?:\.\d)?)", line.strip())
            m_price = re.fullmatch(r"([+-]\d{2,4})", price.strip())
            if not m_line or not m_price:
                continue
            over_price = int(m_price.group(1))
            book = books[i // 2] if i // 2 < len(books) else ""
            entry.quotes.append(BookTotal(
                book=book, total=float(m_line.group(1)), over_price=over_price,
                under_price=estimate_under_price(over_price, hold),
                under_estimated=True,
            ))
        if entry.quotes:
            out.append(entry)

    log.info(
        "parsed %d posted win totals (%d with books disagreeing on the number)",
        len(out), sum(1 for e in out if e.totals_disagree),
    )
    return out


def fetch(url: str = SOURCE_URL, timeout: int = 30, session=None) -> str:
    http = session or requests
    response = http.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"{url} returned HTTP {response.status_code}; the aggregator may have "
            "changed or is blocking. Enter totals by hand in win-totals.yml."
        )
    return response.text


def to_lines(
    scraped: Iterable[ScrapedTotal],
    known_teams: Optional[Sequence[str]] = None,
):
    """Turn scraped rows into priceable lines, shopping each side separately.

    The best over and the best under can sit at different numbers, and taking
    the better number is the one edge here that requires no forecasting at all.
    """
    from ..wintotals import WinTotalLine

    known = set(known_teams or [])
    scraped = list(scraped)

    # Two source rows landing on one team means a bad match, and a bad match
    # invents edge. Drop both rather than guess which one was right.
    seen: Dict[str, int] = {}
    for entry in scraped:
        team = _match(entry.team, known) if known else entry.key
        if team:
            seen[team] = seen.get(team, 0) + 1
    collisions = {t for t, n in seen.items() if n > 1}
    if collisions:
        log.warning(
            "dropping %d team(s) matched by more than one posted total: %s",
            len(collisions), ", ".join(sorted(collisions)),
        )

    lines: List[WinTotalLine] = []
    for entry in scraped:
        over, under = entry.best_over(), entry.best_under()
        if over is None or under is None:
            continue
        team = _match(entry.team, known) if known else entry.key
        if team is None:
            log.info("no rated team matches posted total %r", entry.team)
            continue
        if team in collisions:
            continue
        lines.append(WinTotalLine(
            team=team,
            total=over.total,
            over_price=over.over_price,
            under_price=under.under_price if under.under_price is not None else -110,
            under_total=under.total,
            book=over.book,
            under_book=under.book,
            under_price_estimated=under.under_estimated,
        ))
    return lines


def _match(name: str, known: set) -> Optional[str]:
    """Map the aggregator's team name onto a rated team key.

    Prefix matching alone is actively dangerous here, because one school's
    name is often a prefix of another's. "Louisiana-Monroe" starts with
    "Louisiana ", so a prefix rule silently hands ULM's win total to the Ragin'
    Cajuns — which manufactured a three-win edge out of nothing before this was
    caught. Explicit aliases are checked first, and the fuzzy fallbacks only
    run when no alias applies.
    """
    key = normalize_team(name)
    if key in ALIASES:
        aliased = ALIASES[key]
        return aliased if aliased in known else None
    if key in known:
        return key
    prefixed = [k for k in known if key.startswith(k + " ")]
    if prefixed:
        return max(prefixed, key=len)
    contained = [k for k in known if f" {k} " in f" {key} "]
    return max(contained, key=len) if contained else None


def load(
    known_teams: Optional[Sequence[str]] = None,
    url: str = SOURCE_URL,
    hold: float = DEFAULT_HOLD,
    session=None,
):
    """Fetch, parse and convert in one call. Never fatal."""
    try:
        return to_lines(parse(fetch(url, session=session), hold=hold), known_teams)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not scrape win totals: %s", exc)
        return []
