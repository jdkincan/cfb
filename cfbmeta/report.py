"""Render the weekly readout as HTML and plain text."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__
from .config import Config
from .model import GameProjection
from .probability import MarginDistribution

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

SOURCE_CREDITS = (
    "SP+ (Bill Connelly), ESPN FPI, Elo, SRS and the 247 talent composite, "
    "all via collegefootballdata.com"
)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def format_spread(margin: Optional[float], team: str) -> str:
    """Quote a margin the way a sportsbook would: favorite and a minus number."""
    if margin is None:
        return "—"
    return f"{team} -{abs(margin):.1f}"


def _kickoff_local(proj: GameProjection, tz: ZoneInfo) -> str:
    if not proj.kickoff:
        return ""
    return proj.kickoff.astimezone(tz).strftime("%a %-I:%M %p").replace(" 0", " ")


def _market_line_text(proj: GameProjection) -> str:
    if proj.market_spread is None:
        return "no line"
    mm = -proj.market_spread
    team = proj.home_team if mm >= 0 else proj.away_team
    return f"{team} -{abs(mm):.1f}"


def _side_team(proj: GameProjection) -> str:
    bet = proj.spread_bet
    if not bet or not bet.is_play:
        return ""
    return proj.home_team if bet.side == "home" else proj.away_team


def _ledger(proj: GameProjection) -> List[tuple]:
    """The adjustment audit trail, so every point is traceable."""
    rows: List[tuple] = [("Blended ratings (neutral)", f"{proj.neutral_margin:+.1f}")]

    hfa = proj.hfa or {}
    if proj.neutral_site:
        rows.append(("Home field", "none (neutral site)"))
    elif hfa.get("total"):
        label = f"Home field ({proj.home_team})"
        if hfa.get("elevation", 0) > 0.05:
            label += f", incl. {hfa['elevation']:+.1f} altitude"
        rows.append((label, f"{hfa['total']:+.1f}"))

    coaching = proj.coaching or {}
    if abs(float(coaching.get("total", 0) or 0)) > 0.05:
        rows.append(
            (
                f"Coaching ({coaching.get('home_coach', '?')} vs {coaching.get('away_coach', '?')})",
                f"{coaching['total']:+.1f}",
            )
        )

    avail = proj.availability or {}
    if abs(float(avail.get("total", 0) or 0)) > 0.05:
        reasons = " / ".join(
            r for r in (avail.get("home_reason"), avail.get("away_reason")) if r
        )
        rows.append((f"Availability{' — ' + reasons if reasons else ''}",
                     f"{avail['total']:+.1f}"))

    sit = proj.situational or {}
    rest = sit.get("rest") or {}
    if abs(float(rest.get("total", 0) or 0)) > 0.05:
        rows.append(
            (
                f"Rest ({rest.get('home_days', 0):.0f}d vs {rest.get('away_days', 0):.0f}d)",
                f"{rest['total']:+.1f}",
            )
        )
    travel = sit.get("travel") or {}
    if abs(float(travel.get("total", 0) or 0)) > 0.05:
        rows.append(
            (f"Travel ({abs(travel.get('miles', 0)):,.0f} mi)", f"{travel['total']:+.1f}")
        )

    wx = proj.weather or {}
    if wx.get("available") and (abs(wx.get("spread", 0)) > 0.05 or abs(wx.get("total", 0)) > 0.5):
        rows.append(
            (f"Weather ({wx.get('summary', '')})",
             f"{wx.get('spread', 0):+.1f} spread, {wx.get('total', 0):+.1f} total")
        )

    if proj.market_movement is not None and abs(proj.market_movement) >= 0.5:
        direction = "toward home" if proj.market_movement > 0 else "toward away"
        rows.append(
            ("Line movement since open",
             f"{proj.market_movement:+.1f} ({direction}, open {proj.market_open:+.1f})")
        )
    if proj.best_line is not None and proj.best_book:
        rows.append(("Best number", f"{proj.best_line:+.1f} at {proj.best_book}"))
    # Which price the edge was measured against. A sharp book's number is a
    # fair price; a median of retail books is a guess about a fair price, and
    # beating it can just mean those books have not moved yet.
    if proj.market_provider:
        rows.append(("Benchmark", proj.market_provider))

    if proj.projected_total is not None:
        rows.append(
            (
                "Projected score",
                f"{proj.home_team} {proj.projected_home_points:.0f}"
                f" - {proj.projected_away_points:.0f} {proj.away_team}",
            )
        )
    if proj.market_total is not None and proj.projected_total is not None:
        rows.append(
            ("Total (model vs market)", f"{proj.projected_total:.1f} vs {proj.market_total:.1f}")
        )
    return rows


def _view(proj: GameProjection, tz: ZoneInfo) -> Dict[str, Any]:
    """Flatten a projection into exactly what the template needs."""
    bet = proj.spread_bet
    is_play = bool(bet and bet.is_play)
    view: Dict[str, Any] = {
        "matchup": proj.matchup,
        "home_team": proj.home_team,
        "away_team": proj.away_team,
        "neutral_site": proj.neutral_site,
        "venue": proj.venue,
        "kickoff_local": _kickoff_local(proj, tz),
        "projected_line": proj.projected_line,
        "projected_margin": proj.projected_margin,
        "market_line_text": _market_line_text(proj),
        "edge_text": "—" if proj.edge is None else f"{proj.edge:+.1f}",
        "home_win_probability": proj.home_win_probability,
        "is_play": is_play,
        "components": proj.components,
        "ledger": _ledger(proj),
        "notes": proj.notes,
        "confidence": proj.confidence,
    }
    if bet:
        view["spread_bet"] = {
            "side": bet.side,
            "side_team": _side_team(proj),
            "line": bet.line,
            "edge_points": bet.edge_points,
            "cover_probability": bet.cover_probability,
            "push_probability": bet.push_probability,
            "expected_value": bet.expected_value,
            "units": bet.units,
        }
    return view


def _validation_banner() -> Dict[str, Any]:
    """State plainly whether the model has ever beaten the market fairly."""
    from .backtest import is_validated, load_result

    record = load_result()
    if is_validated(record):
        return {"validated": True, "text": ""}

    if not record:
        detail = "no backtest has been run yet."
    else:
        basis = record.get("basis", "?")
        mae, market = record.get("mae"), record.get("market_mae")
        ats = record.get("ats_pct")
        parts = [f"the only backtest so far used a '{basis}' basis"]
        if mae is not None and market is not None:
            parts.append(f"MAE {mae:.2f} vs the closing line's {market:.2f}")
        if ats is not None:
            parts.append(f"ATS {ats:.1%} against a 52.4% break-even")
        detail = "; ".join(parts) + "."
    return {
        "validated": False,
        "text": (
            "This model has not been shown to beat the closing line: " + detail +
            " A fair test needs point-in-time ratings, which only accumulate as "
            "weekly snapshots are archived. Treat the stakes below as a record of "
            "what the model thinks, not as advice to bet."
        ),
    }


def _spotlight_view(spot, tz) -> Optional[Dict[str, Any]]:
    if spot is None:
        return None
    proj = spot.projection

    def side(detail, is_home):
        return {
            "team": detail.team,
            "role": "home" if is_home else "away",
            "coach": detail.coach,
            "coach_score": detail.coach_score,
            "talent": round(detail.talent, 2) if detail.talent is not None else None,
            "blue_chip": (
                f"{detail.blue_chip_ratio:.0%}" if detail.blue_chip_ratio is not None else "—"
            ),
            "players": [
                {"name": p.name, "position": p.position, "class_year": p.class_year,
                 "stars": p.stars_label, "rating": f"{p.rating:.4f}" if p.rating else "—"}
                for p in detail.players
            ],
            "momentum": detail.momentum,
            "momentum_delta": detail.momentum_delta,
            "momentum_recent": detail.momentum_recent,
        }

    view = _view(proj, tz)
    view["sides"] = [side(spot.away, False), side(spot.home, True)]
    view["note"] = spot.note
    return view


def build_context(
    projections: Sequence[GameProjection],
    config: Config,
    week: int,
    season: int,
    generated_at: Optional[dt.datetime] = None,
    book=None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tz = ZoneInfo(config.timezone)
    generated_at = generated_at or dt.datetime.now(tz)

    views = [_view(p, tz) for p in projections]
    plays = [v for v in views if v["is_play"]]
    total_units = sum(v["spread_bet"]["units"] for v in plays)

    # A worked example for the footer, so the edge numbers stay in perspective.
    example_edge = 3.0
    example_cover = MarginDistribution(example_edge, config.sigma_margin).p_home_cover(0.0)

    extras = extras or {}
    validation = _validation_banner()
    spplus = extras.get("spplus_check")

    return {
        "validation": validation,
        "spplus_check": (
            {"ok": spplus.ok, "text": spplus.describe(),
             "url": config.spplus_article_url,
             "mismatched": spplus.mismatched}
            if spplus is not None else None
        ),
        "spotlight": _spotlight_view(extras.get("spotlight"), tz),
        "title": f"{config.email_subject_prefix} — Week {week}",
        "subtitle": generated_at.strftime("%A, %B %-d, %Y").replace(" 0", " ")
        + f" · {season} season",
        "summary_tiles": [
            ("Games", len(views)),
            ("Plays", len(plays)),
            ("Units", f"{total_units:.1f}"),
            ("Avg edge", f"{(sum(abs(p['spread_bet']['edge_points']) for p in plays) / len(plays)):.1f}" if plays else "—"),
        ],
        "plays": plays,
        "games": views,
        "detail_games": views,
        "min_edge": config.min_edge_points,
        "strong_edge": config.strong_edge_points,
        "sigma": config.sigma_margin,
        "example_edge": example_edge,
        "example_cover": example_cover,
        "method_line": (
            "Each rating system proposes a neutral-field margin; those are blended by "
            "weight, then venue-specific home field, coaching and rest/travel "
            "adjustments are added on top. The market is the benchmark, not an input."
        ),
        "sources_line": SOURCE_CREDITS,
        # Which sources actually had data for this season, and which didn't.
        # Preseason runs lean on fewer sources and the readout must say so.
        "provenance": book.provenance_lines() if book is not None else [],
        "usable_sources": book.usable_sources() if book is not None else [],
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M %Z"),
        "version": f"cfbmeta {__version__}",
        "week": week,
        "season": season,
    }


def render_html(context: Dict[str, Any]) -> str:
    return _env().get_template("email.html.j2").render(**context)


def render_text(context: Dict[str, Any]) -> str:
    """Plain-text alternative for clients that won't render HTML."""
    lines = [context["title"], context["subtitle"], ""]

    banner = context.get("validation") or {}
    if not banner.get("validated") and banner.get("text"):
        lines += ["!" * 78, "UNVALIDATED — " + banner["text"], "!" * 78, ""]

    spot = context.get("spotlight")
    if spot:
        lines.append(f"SPOTLIGHT — {spot['matchup']}")
        lines.append("-" * 78)
        lines.append(f"  model {spot['projected_line']}   market {spot['market_line_text']}"
                     f"   edge {spot['edge_text']}")
        for s_ in spot["sides"]:
            head = f"  {s_['team']} ({s_['role']})"
            if s_["coach"]:
                head += f" — {s_['coach']} {s_['coach_score']:+.2f}"
            lines.append(head)
            lines.append(f"     talent {s_['talent']}  blue-chip {s_['blue_chip']}")
            for p in s_["players"]:
                lines.append(f"     {p['stars']} {p['name']:<24}{p['position']:<4}"
                             f"{p['class_year']:<4}{p['rating']}")
            if s_["momentum_delta"] is not None:
                lines.append(f"     SP+ momentum: {s_['momentum_delta']:+.2f} since week "
                             f"{s_['momentum'][0][0]}, {s_['momentum_recent']:+.2f} last week")
        if spot.get("note"):
            lines.append(f"  {spot['note']}")
        lines.append("")

    if context["plays"]:
        lines.append(f"BEST BETS ({len(context['plays'])})")
        lines.append("-" * 60)
        for play in context["plays"]:
            bet = play["spread_bet"]
            lines.append(
                f"  {bet['side_team']} {bet['line']:+.1f}  ({bet['units']:.2f}u)"
            )
            lines.append(f"    {play['matchup']}")
            lines.append(
                f"    model {play['projected_line']} vs market {play['market_line_text']}"
                f"  edge {bet['edge_points']:+.1f}"
                f"  cover {bet['cover_probability'] * 100:.1f}%"
            )
            lines.append("")
    else:
        lines.append(
            f"No game cleared the {context['min_edge']:.1f}-point edge threshold."
        )
        lines.append("")

    lines.append(f"FULL SLATE ({len(context['games'])} games)")
    lines.append("-" * 78)
    lines.append(f"{'Game':<34}  {'Model':>18}  {'Market':>18}  {'Edge':>5}")
    for game in context["games"]:
        lines.append(
            f"{game['matchup'][:34]:<34}  {game['projected_line'][:18]:>18}  "
            f"{game['market_line_text'][:18]:>18}  {game['edge_text']:>5}"
        )

    lines += ["", "-" * 78, context["method_line"], ""]
    if context.get("provenance"):
        lines.append(f"DATA (season {context['season']})")
        lines += [f"  {line}" for line in context["provenance"]]
        lines.append("")
    check = context.get("spplus_check")
    if check:
        lines.append(("OK   " if check["ok"] else "WARN ") + check["text"])
        lines.append(f"  vs {check['url']}")
        lines.append("")
    lines += [f"Sources: {context['sources_line']}",
              f"Generated {context['generated_at']} · {context['version']}"]
    return "\n".join(lines)


def render(
    projections: Sequence[GameProjection],
    config: Config,
    week: int,
    season: int,
    generated_at: Optional[dt.datetime] = None,
    book=None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    context = build_context(projections, config, week, season, generated_at, book, extras)
    return {
        "subject": f"{config.email_subject_prefix} — Week {week} ({len(context['plays'])} plays)",
        "html": render_html(context),
        "text": render_text(context),
    }
