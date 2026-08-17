"""Track whether a rating source is actually changing between runs.

The source audit already catches a source that is missing and a source that is
flat. It could not catch the third failure: one serving real, well-dispersed,
*stale* numbers — a feed that stopped updating, or a provider revision that
never propagated. Those look perfectly healthy from a single run.

The only way to see it is across runs, so each run records a content hash per
source and the date that hash first appeared. If SP+ has served identical
values for two weeks in October, something upstream is wrong, and the readout
should say so rather than quietly projecting on last month's ratings.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

STATE_PATH = Path(__file__).resolve().parent.parent / "calibration" / "source_state.json"


def fingerprint(values: Dict[str, float]) -> str:
    """Stable hash of a source's ratings, insensitive to dict ordering."""
    payload = ";".join(f"{k}={values[k]:.4f}" for k in sorted(values))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_state(path: Optional[Path] = None) -> Dict[str, dict]:
    path = path or STATE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: Dict[str, dict], path: Optional[Path] = None) -> None:
    path = path or STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True))
    except OSError as exc:  # best effort; never fatal
        log.debug("could not write source state: %s", exc)


def record(
    key: str,
    digest: str,
    state: Dict[str, dict],
    now: Optional[dt.datetime] = None,
) -> float:
    """Update state for one source; return days since these values first appeared."""
    now = now or dt.datetime.now(dt.timezone.utc)
    entry = state.get(key)
    if entry and entry.get("fingerprint") == digest:
        try:
            first = dt.datetime.fromisoformat(entry["first_seen"])
        except (KeyError, ValueError):
            first = now
        if first.tzinfo is None:
            first = first.replace(tzinfo=dt.timezone.utc)
        entry["last_seen"] = now.isoformat()
        return max(0.0, (now - first).total_seconds() / 86400.0)

    state[key] = {
        "fingerprint": digest,
        "first_seen": now.isoformat(),
        "last_seen": now.isoformat(),
    }
    return 0.0
