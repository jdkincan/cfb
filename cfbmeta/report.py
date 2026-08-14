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


def build_context(
    projections: Sequence[GameProjection],
    config: Config,
    week: int,
    season: int,
    generated_at: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    tz = ZoneInfo(config.timezone)
    generated_at = generated_at or dt.datetime.now(tz)

    views = [_view(p, tz) for p in projections]
    plays = [v for v in views if v["is_play"]]
    total_units = sum(v["spread_bet"]["units"] for v in plays)

    # A worked example for the footer, so the edge numbers stay in perspective.
    example_edge = 3.0
    example_cover = MarginDistribution(example_edge, config.sigma_margin).p_home_cover(0.0)

    return {
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

    lines += ["", "-" * 60, context["method_line"], "", f"Sources: {context['sources_line']}",
              f"Generated {context['generated_at']} · {context['version']}"]
    return "\n".join(lines)


def render(
    projections: Sequence[GameProjection],
    config: Config,
    week: int,
    season: int,
    generated_at: Optional[dt.datetime] = None,
) -> Dict[str, str]:
    context = build_context(projections, config, week, season, generated_at)
    return {
        "subject": f"{config.email_subject_prefix} — Week {week} ({len(context['plays'])} plays)",
        "html": render_html(context),
        "text": render_text(context),
    }
