"""Fallback FPI reader, straight from ESPN.

FPI normally arrives through CFBD's ``/ratings/fpi``, which mirrors ESPN on its
own ingest schedule. Early in a season those can diverge: ESPN publishes the
new season's FPI before CFBD has picked it up. This module covers that gap so a
lag upstream doesn't silently cost a whole rating source.

CFBD stays primary. This runs only when CFBD's FPI comes back unusable, and it
only fills in teams the book already knows about, so a name that doesn't join
cleanly is skipped rather than inventing a phantom team.

A note on the parsing. ESPN's power-index endpoint is undocumented and its
nesting has changed shape over the years, so rather than betting on one exact
path, :func:`extract_fpi_rows` walks the payload looking for a team name and a
value labelled as FPI wherever they happen to live. That is deliberately more
tolerant than a precise parser: the failure mode of a wrong guess here is a
silently missing source, and tolerance is cheaper than precision when the
payload cannot be verified in advance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

FPI_URL = (
    "https://site.web.api.espn.com/apis/fitt/v3/sports/football/"
    "college-football/powerindex"
)

# Labels ESPN has used for the overall FPI number.
_FPI_LABELS = {"fpi", "overall", "total", "fpioverall"}
# Labels that look like FPI but are ranks or projections, not the rating.
_REJECT_LABELS = {"rank", "ranking", "proj", "projected", "wins", "losses", "sos"}


def _clean(text: Any) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # FPI lives roughly in [-40, 40]. Anything wilder is a rank, a win total or
    # a percentage that happened to sit under a matching label.
    return result if -100.0 < result < 100.0 else None


def _find_team_name(blob: Any) -> Optional[str]:
    if not isinstance(blob, dict):
        return None
    team = blob.get("team") if isinstance(blob.get("team"), dict) else blob
    for key in ("displayName", "location", "nickname", "name", "shortDisplayName", "abbreviation"):
        value = team.get(key) if isinstance(team, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _find_fpi_value(blob: Any, depth: int = 0) -> Optional[float]:
    """Search a team blob for the overall FPI rating, whatever the nesting."""
    if depth > 6:
        return None

    if isinstance(blob, dict):
        # A stat object: something naming it FPI plus a value alongside.
        label = " ".join(
            str(blob.get(k, "")) for k in ("name", "shortDisplayName", "displayName", "abbreviation")
        )
        cleaned = _clean(label)
        if cleaned in _FPI_LABELS or cleaned.startswith("fpi"):
            if not any(bad in cleaned for bad in _REJECT_LABELS):
                for value_key in ("value", "displayValue", "rating"):
                    found = _as_float(blob.get(value_key))
                    if found is not None:
                        return found

        # A plain `fpi: 12.3` on the object itself.
        direct = _as_float(blob.get("fpi"))
        if direct is not None:
            return direct

        for key, value in blob.items():
            if _clean(key) in _REJECT_LABELS:
                continue
            found = _find_fpi_value(value, depth + 1)
            if found is not None:
                return found

    elif isinstance(blob, list):
        for item in blob:
            found = _find_fpi_value(item, depth + 1)
            if found is not None:
                return found

    return None


def extract_fpi_rows(payload: Any) -> List[Dict[str, Any]]:
    """Pull ``[{"team": name, "fpi": rating}, ...]`` out of an ESPN payload.

    Pure and network-free, so the parsing is testable without ESPN.
    """
    entries: List[Any] = []
    if isinstance(payload, dict):
        for key in ("teams", "items", "entries", "standings"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                entries = value
                break
        if not entries:
            for value in payload.values():
                if isinstance(value, dict):
                    nested = extract_fpi_rows(value)
                    if nested:
                        return nested
    elif isinstance(payload, list):
        entries = payload

    rows: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        name = _find_team_name(entry)
        fpi = _find_fpi_value(entry)
        if not name or fpi is None or name in seen:
            continue
        seen.add(name)
        rows.append({"team": name, "fpi": fpi})
    return rows


def fetch_fpi(
    season: int,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch this season's FPI from ESPN. Returns [] on any failure."""
    params = {"region": "us", "lang": "en", "limit": 200, "season": season}
    http = session or requests
    try:
        response = http.get(
            FPI_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "cfbmeta/1.0", "Accept": "application/json"},
        )
        if response.status_code != 200:
            log.warning("ESPN FPI returned HTTP %s", response.status_code)
            return []
        rows = extract_fpi_rows(response.json())
    except (requests.RequestException, ValueError) as exc:
        log.warning("ESPN FPI fetch failed: %s", exc)
        return []

    log.info("ESPN FPI: parsed %d teams for %d", len(rows), season)
    return rows
