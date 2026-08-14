"""Client for the collegefootballdata.com API.

Written defensively on purpose. CFBD has shipped both snake_case (v1) and
camelCase (v2) payloads and has moved a few endpoints between path prefixes,
so rather than betting on one shape:

* :func:`pick` looks a field up under several spellings.
* Each logical endpoint carries a list of candidate paths and the first one
  that answers is remembered for the rest of the process.

``cfbmeta doctor`` exercises every endpoint and prints what actually came back,
which is the fastest way to confirm the mapping against the live API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

log = logging.getLogger(__name__)

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _snake(name: str) -> str:
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _variants(name: str) -> List[str]:
    """All the spellings a single logical field might arrive under."""
    snake = _snake(name)
    parts = snake.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    pascal = "".join(p.title() for p in parts)
    seen, out = set(), []
    for cand in (name, snake, camel, pascal, snake.replace("_", "")):
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def pick(obj: Any, *names: str, default: Any = None) -> Any:
    """Fetch the first present field from ``obj`` trying several spellings.

    Dotted names walk nested objects: ``pick(row, "offense.rating")``.
    """
    if obj is None:
        return default
    for name in names:
        if "." in name:
            cur: Any = obj
            for part in name.split("."):
                cur = pick(cur, part, default=None)
                if cur is None:
                    break
            if cur is not None:
                return cur
            continue
        if isinstance(obj, dict):
            for cand in _variants(name):
                if cand in obj and obj[cand] is not None:
                    return obj[cand]
            # Last resort: case-insensitive, punctuation-insensitive match.
            flat = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in obj.items()}
            key = re.sub(r"[^a-z0-9]", "", _snake(name).replace("_", ""))
            if flat.get(key) is not None:
                return flat[key]
        else:
            for cand in _variants(name):
                if hasattr(obj, cand):
                    val = getattr(obj, cand)
                    if val is not None:
                        return val
    return default


def pick_float(obj: Any, *names: str, default: Optional[float] = None) -> Optional[float]:
    val = pick(obj, *names, default=None)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class CFBDError(RuntimeError):
    pass


class CFBDAuthError(CFBDError):
    pass


# Logical endpoint -> candidate paths, most likely first.
ENDPOINTS: Dict[str, Sequence[str]] = {
    "calendar": ("/calendar",),
    "games": ("/games",),
    "lines": ("/lines",),
    "teams": ("/teams/fbs", "/teams"),
    "venues": ("/venues",),
    "sp": ("/ratings/sp",),
    "srs": ("/ratings/srs",),
    "elo": ("/ratings/elo",),
    "fpi": ("/ratings/fpi",),
    "talent": ("/talent",),
    "returning": ("/player/returning", "/players/returning"),
    "coaches": ("/coaches",),
    "ppa_teams": ("/ppa/teams", "/metrics/ppa/teams"),
    "advanced_stats": ("/stats/season/advanced",),
}


class CFBDClient:
    """Thin, cached, retrying wrapper over the CFBD REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.collegefootballdata.com",
        timeout: int = 30,
        max_retries: int = 4,
        cache_dir: Optional[str] = "data/cache",
        cache_ttl_minutes: int = 360,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CFBD_API_KEY") or ""
        if not self.api_key:
            raise CFBDAuthError(
                "No CFBD API key. Set CFBD_API_KEY (free key: "
                "https://collegefootballdata.com/key)."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl_minutes * 60
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "cfbmeta/1.0",
            }
        )
        self._resolved_paths: Dict[str, str] = {}

    # -- plumbing ------------------------------------------------------------
    def _cache_path(self, path: str, params: Dict[str, Any]) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = path.strip("/").replace("/", "_")
        parts = "_".join(f"{k}-{v}" for k, v in sorted(params.items()) if v is not None)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "", f"{key}__{parts}") or key
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, cache_file: Optional[Path]) -> Optional[Any]:
        if not cache_file or not cache_file.exists():
            return None
        if self.cache_ttl and (time.time() - cache_file.stat().st_mtime) > self.cache_ttl:
            return None
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, cache_file: Optional[Path], payload: Any) -> None:
        if not cache_file:
            return
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload))
        except OSError as exc:  # cache is best-effort, never fatal
            log.debug("cache write failed for %s: %s", cache_file, exc)

    def _request(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        clean = {k: v for k, v in params.items() if v is not None}
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=clean, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                self._sleep(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise CFBDError(f"{path} returned non-JSON: {resp.text[:200]}") from exc
            if resp.status_code in (401, 403):
                raise CFBDAuthError(
                    f"{path} rejected the API key ({resp.status_code}). "
                    "Check CFBD_API_KEY, and that your tier allows this endpoint."
                )
            if resp.status_code == 404:
                raise FileNotFoundError(path)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = CFBDError(f"{path} -> HTTP {resp.status_code}")
                retry_after = resp.headers.get("Retry-After")
                self._sleep(attempt, float(retry_after) if retry_after else None)
                continue
            raise CFBDError(f"{path} -> HTTP {resp.status_code}: {resp.text[:200]}")

        raise CFBDError(f"{path} failed after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _sleep(attempt: int, override: Optional[float] = None) -> None:
        time.sleep(override if override is not None else min(2 ** attempt, 16))

    def get(self, endpoint: str, use_cache: bool = True, **params: Any) -> List[Dict[str, Any]]:
        """Call a logical endpoint, trying each candidate path until one works."""
        candidates = (
            [self._resolved_paths[endpoint]]
            if endpoint in self._resolved_paths
            else list(ENDPOINTS.get(endpoint, (endpoint,)))
        )

        errors: List[str] = []
        for path in candidates:
            cache_file = self._cache_path(path, params) if use_cache else None
            cached = self._read_cache(cache_file)
            if cached is not None:
                self._resolved_paths[endpoint] = path
                return cached
            try:
                payload = self._request(path, params)
            except FileNotFoundError:
                errors.append(f"{path}: 404")
                continue
            self._resolved_paths[endpoint] = path
            if isinstance(payload, dict):
                payload = [payload]
            self._write_cache(cache_file, payload)
            return payload

        raise CFBDError(
            f"no working path for '{endpoint}'. Tried: {'; '.join(errors) or candidates}"
        )

    # -- typed-ish accessors -------------------------------------------------
    def calendar(self, year: int) -> List[Dict[str, Any]]:
        return self.get("calendar", year=year)

    def games(
        self, year: int, week: Optional[int] = None, season_type: str = "regular"
    ) -> List[Dict[str, Any]]:
        return self.get("games", year=year, week=week, seasonType=season_type)

    def lines(
        self, year: int, week: Optional[int] = None, season_type: str = "regular"
    ) -> List[Dict[str, Any]]:
        # Lines move all week; keep this cache short.
        return self.get("lines", year=year, week=week, seasonType=season_type, use_cache=False)

    def teams(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.get("teams", year=year)

    def venues(self) -> List[Dict[str, Any]]:
        return self.get("venues")

    def sp_ratings(self, year: int) -> List[Dict[str, Any]]:
        return self.get("sp", year=year)

    def srs_ratings(self, year: int) -> List[Dict[str, Any]]:
        return self.get("srs", year=year)

    def elo_ratings(self, year: int, week: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.get("elo", year=year, week=week)

    def fpi_ratings(self, year: int) -> List[Dict[str, Any]]:
        return self.get("fpi", year=year)

    def talent(self, year: int) -> List[Dict[str, Any]]:
        return self.get("talent", year=year)

    def returning_production(self, year: int) -> List[Dict[str, Any]]:
        return self.get("returning", year=year)

    def coaches(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.get("coaches", year=year)

    def ppa_teams(self, year: int) -> List[Dict[str, Any]]:
        return self.get("ppa_teams", year=year)


def iter_rows(payload: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    for row in payload or []:
        if isinstance(row, dict):
            yield row
