"""Verify CFBD's SP+ against what Bill Connelly actually published.

CFBD mirrors SP+ rather than producing it, so it can lag a revision without any
outward sign — the payload carries no version or as-of field. The freshness
tracker can tell that values haven't moved; it cannot tell whether they *should*
have. Only the source of record settles that.

ESPN's article HTML sits behind an AWS WAF challenge and cannot be scraped, but
their content API serves the same story as JSON and is reachable::

    https://now.core.api.espn.com/v1/sports/news/<article-id>

The rankings themselves live in an embedded widget that the API does not
return. What the prose does contain is a set of statements like "Oklahoma State
(up 22.0 points, 39th overall)" — year-over-year deltas with the new rank. That
is enough: those deltas are computable from CFBD's own two seasons, so each one
is an independent check on whether CFBD holds the published numbers.

Run against the 2026 final preseason article this matched 15 of 15 deltas
exactly, which is how we know the 2026 ratings are current rather than stale.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

CONTENT_API = "https://now.core.api.espn.com/v1/sports/news/{article_id}"

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_ORDINAL_PATTERN = "|".join(list(_ORDINALS) + [r"\d{1,3}(?:st|nd|rd|th)"])
# Case-sensitive on purpose. A team name is Capitalised words, and IGNORECASE
# would make [A-Z] match lowercase too — which silently swallowed the whole
# preceding sentence as the "team name" and cost half the checks.
# Team names run 1-4 words ("UConn", "Ole Miss", "Miami (OH)"), so the
# repetition is bounded rather than greedy.
_CLAIM = re.compile(
    r"\b([A-Z][\w'&.\-]*(?:\s[A-Z][\w'&.\-]*){0,3})\s*\(\s*"
    r"(?:[Uu]p|[Dd]own)\s+([\d.]+)\s+points?,\s*"
    rf"({_ORDINAL_PATTERN})\s+overall\s*\)"
)
_DIRECTION = re.compile(r"\(\s*([Uu]p|[Dd]own)\s")


@dataclass
class Claim:
    team: str
    delta: float          # published year-over-year change, signed
    rank: int


@dataclass
class CheckResult:
    article_id: str = ""
    headline: str = ""
    published: str = ""
    claims: List[Claim] = field(default_factory=list)
    matched: int = 0
    mismatched: List[Tuple[str, float, float]] = field(default_factory=list)
    error: str = ""

    @property
    def checked(self) -> int:
        return self.matched + len(self.mismatched)

    @property
    def ok(self) -> bool:
        """Every published delta we could check reproduced from CFBD."""
        return self.checked > 0 and not self.mismatched

    def describe(self) -> str:
        if self.error:
            return f"SP+ publication check unavailable: {self.error}"
        if not self.checked:
            return "SP+ publication check: no comparable figures found in the article"
        head = (
            f"SP+ matches ESPN publication ({self.matched}/{self.checked} deltas"
            if self.ok
            else f"SP+ DISAGREES with ESPN ({self.matched}/{self.checked} deltas match"
        )
        return f"{head}, published {self.published[:10]})"


def _ordinal_to_int(text: str) -> Optional[int]:
    text = text.strip().lower()
    if text in _ORDINALS:
        return _ORDINALS[text]
    digits = re.match(r"(\d{1,3})", text)
    return int(digits.group(1)) if digits else None


def parse_claims(story_html: str) -> List[Claim]:
    """Pull "(up 22.0 points, 39th overall)" statements out of the article."""
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", story_html or "")))
    claims: List[Claim] = []
    seen = set()
    for match in _CLAIM.finditer(text):
        team, amount, ordinal = match.groups()
        rank = _ordinal_to_int(ordinal)
        team = team.strip()
        direction = _DIRECTION.search(match.group(0))
        if rank is None or team in seen or direction is None:
            continue
        seen.add(team)
        delta = float(amount) * (1 if direction.group(1).lower() == "up" else -1)
        claims.append(Claim(team=team, delta=delta, rank=rank))
    return claims


def fetch_article(article_id: str, timeout: int = 30, session=None) -> dict:
    http = session or requests
    response = http.get(
        CONTENT_API.format(article_id=article_id),
        timeout=timeout,
        headers={"User-Agent": "cfbmeta/1.0", "Accept": "application/json"},
    )
    response.raise_for_status()
    payload = response.json()
    headlines = payload.get("headlines") or []
    if not headlines:
        raise ValueError("no article body in the response")
    return headlines[0]


def check(
    article_id: str,
    current: Dict[str, float],
    previous: Dict[str, float],
    tolerance: float = 0.15,
    session=None,
) -> CheckResult:
    """Compare published year-over-year deltas against CFBD's two seasons.

    ``current`` and ``previous`` map a normalized team key to that season's SP+
    rating. Tolerance covers the article rounding to one decimal.
    """
    from ..ratings import normalize_team

    result = CheckResult(article_id=str(article_id))
    try:
        article = fetch_article(article_id, session=session)
    except Exception as exc:  # noqa: BLE001 - a failed check must never break a run
        result.error = f"{type(exc).__name__}: {exc}"
        log.warning("SP+ publication check failed: %s", result.error)
        return result

    result.headline = article.get("headline", "")
    result.published = article.get("published", "")
    result.claims = parse_claims(article.get("story", ""))

    for claim in result.claims:
        key = normalize_team(claim.team)
        if key not in current or key not in previous:
            continue
        actual = current[key] - previous[key]
        if abs(actual - claim.delta) <= tolerance:
            result.matched += 1
        else:
            result.mismatched.append((claim.team, claim.delta, round(actual, 2)))

    if result.mismatched:
        log.warning(
            "SP+ disagrees with the published article on %d team(s): %s",
            len(result.mismatched),
            ", ".join(f"{t} pub {p:+.1f} vs cfbd {a:+.1f}" for t, p, a in result.mismatched[:5]),
        )
    else:
        log.info("SP+ publication check: %s", result.describe())
    return result
