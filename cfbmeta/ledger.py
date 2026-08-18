"""What was actually bet, at what price, and how it settled.

The archive records what the model *recommended*. This records what you
*did* — which is a different thing, and the difference is where most of the
honest accounting lives. Recommended stakes are frictionless; real ones have a
price you got rather than the one you modelled, a book that moved before you
clicked, and the plays you passed on.

Two numbers only become computable once this exists:

* **Realised ROI** against units actually risked, not units suggested.
* **Closing line value** — the number you got against the number the market
  closed at. CLV is the fastest honest read on whether an edge is real. Results
  need hundreds of bets to separate skill from variance; CLV needs dozens,
  because beating the close is the thing a genuine edge does *mechanically*.

The file is plain CSV so it can be edited by hand between runs, which is how it
will actually be maintained.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).resolve().parent.parent / "archive" / "bets.csv"

COLUMNS = [
    "placed_at", "season", "week", "game_id", "matchup", "side", "team",
    "line_taken", "price", "units", "book",
    "model_margin", "model_edge", "consensus_at_bet",
    "closing_line", "clv_points", "result", "profit_units", "notes",
]


@dataclass
class Bet:
    season: int
    week: int
    game_id: str
    matchup: str
    side: str            # home / away / over / under
    team: str
    line_taken: float
    units: float
    price: int = -110
    book: str = ""
    placed_at: str = ""
    model_margin: Optional[float] = None
    model_edge: Optional[float] = None
    consensus_at_bet: Optional[float] = None
    closing_line: Optional[float] = None
    clv_points: Optional[float] = None
    result: str = ""     # win / loss / push / void
    profit_units: Optional[float] = None
    notes: str = ""

    def row(self) -> Dict[str, Any]:
        data = asdict(self)
        data["placed_at"] = self.placed_at or dt.datetime.now(dt.timezone.utc).isoformat()
        return {k: ("" if data.get(k) is None else data.get(k)) for k in COLUMNS}


def load(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path or LEDGER_PATH
    if not path.exists():
        return []
    try:
        with path.open() as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        log.warning("could not read the ledger: %s", exc)
        return []


def append(bets: Sequence[Bet], path: Optional[Path] = None) -> int:
    """Add bets, skipping any already recorded for the same game and side."""
    path = path or LEDGER_PATH
    existing = load(path)
    seen = {(r.get("game_id"), r.get("side")) for r in existing}

    fresh = [b for b in bets if (str(b.game_id), b.side) not in seen]
    if not fresh:
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for bet in fresh:
            writer.writerow(bet.row())
    log.info("recorded %d bet(s) to %s", len(fresh), path)
    return len(fresh)


def _payout(price: int) -> float:
    return (price / 100.0) if price > 0 else (100.0 / abs(price))


def settle(
    results: Dict[str, float],
    closing: Optional[Dict[str, float]] = None,
    path: Optional[Path] = None,
) -> int:
    """Grade open bets from final margins, and record closing line value.

    ``results`` maps game id to the actual home margin. ``closing`` maps game id
    to the closing spread (home side), used for CLV.
    """
    path = path or LEDGER_PATH
    rows = load(path)
    if not rows:
        return 0

    closing = closing or {}
    settled = 0
    for row in rows:
        if row.get("result"):
            continue
        gid = str(row.get("game_id"))
        if gid not in results:
            continue
        margin = float(results[gid])
        try:
            line = float(row["line_taken"])
            units = float(row["units"])
            price = int(row.get("price") or -110)
        except (TypeError, ValueError, KeyError):
            continue

        side = (row.get("side") or "").lower()
        # line_taken is from the bettor's side: home +3 means home covers if
        # margin > -3; away -7 means away covers if margin < 7... expressed
        # against the home margin, the bettor's threshold is always -line.
        threshold = -line
        if side == "home":
            outcome = "push" if margin == threshold else ("win" if margin > threshold else "loss")
        elif side == "away":
            outcome = "push" if margin == threshold else ("win" if margin < threshold else "loss")
        else:
            continue

        row["result"] = outcome
        row["profit_units"] = round(
            0.0 if outcome == "push" else (units * _payout(price) if outcome == "win" else -units),
            3,
        )
        if gid in closing:
            close = float(closing[gid])
            # CLV in points: how much better the number taken was than the
            # close, from the bettor's side.
            row["closing_line"] = close
            row["clv_points"] = round(
                (line - close) if side == "home" else (close - line), 2
            )
        settled += 1

    if settled:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    log.info("settled %d bet(s)", settled)
    return settled


@dataclass
class LedgerSummary:
    bets: int = 0
    settled: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    units_risked: float = 0.0
    profit_units: float = 0.0
    clv_sample: int = 0
    clv_mean: Optional[float] = None
    clv_positive: int = 0

    @property
    def roi(self) -> Optional[float]:
        return self.profit_units / self.units_risked if self.units_risked else None

    @property
    def win_rate(self) -> Optional[float]:
        graded = self.wins + self.losses
        return self.wins / graded if graded else None

    def describe(self) -> str:
        lines = [f"{self.bets} bets recorded, {self.settled} settled"]
        if self.settled:
            lines.append(
                f"  {self.wins}-{self.losses}-{self.pushes}"
                + (f" ({self.win_rate:.1%})" if self.win_rate is not None else "")
            )
            lines.append(
                f"  {self.profit_units:+.2f}u on {self.units_risked:.2f}u risked"
                + (f" ({self.roi:+.1%} ROI)" if self.roi is not None else "")
            )
        if self.clv_sample:
            lines.append(
                f"  CLV {self.clv_mean:+.2f} pts over {self.clv_sample} bets, "
                f"{self.clv_positive}/{self.clv_sample} beat the close"
            )
            lines.append(
                "  CLV is the earliest honest read on edge — results need hundreds "
                "of bets, this needs dozens."
            )
        return "\n".join(lines)


def summarize(path: Optional[Path] = None) -> LedgerSummary:
    rows = load(path)
    summary = LedgerSummary(bets=len(rows))
    clv_values: List[float] = []

    for row in rows:
        result = (row.get("result") or "").lower()
        if result in ("win", "loss", "push"):
            summary.settled += 1
            summary.wins += result == "win"
            summary.losses += result == "loss"
            summary.pushes += result == "push"
            try:
                summary.units_risked += float(row.get("units") or 0)
                summary.profit_units += float(row.get("profit_units") or 0)
            except (TypeError, ValueError):
                pass
        try:
            if row.get("clv_points") not in (None, ""):
                clv_values.append(float(row["clv_points"]))
        except (TypeError, ValueError):
            pass

    if clv_values:
        summary.clv_sample = len(clv_values)
        summary.clv_mean = round(sum(clv_values) / len(clv_values), 3)
        summary.clv_positive = sum(1 for v in clv_values if v > 0)
    summary.profit_units = round(summary.profit_units, 3)
    summary.units_risked = round(summary.units_risked, 3)
    return summary
