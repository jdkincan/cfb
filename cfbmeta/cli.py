"""Command line entry point.

    python -m cfbmeta run          # the weekly job: build, render, email
    python -m cfbmeta preview      # same, written to a file instead of sent
    python -m cfbmeta doctor       # verify credentials and every API endpoint
    python -m cfbmeta backtest     # measure accuracy, optionally refit weights
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo

from .adjustments import build_situational_model, parse_start
from .backtest import collect_rows, evaluate, fit_edge_shrink, fit_key_numbers, fit_weights
from .coaching import build_coach_model
from .config import Config, load_dotenv
from .hfa import build_hfa_model
from .model import project_slate
from .probability import KEY_NUMBERS_PATH
from .ratings import build_rating_book
from .report import render
from .sources.cfbd import CFBDClient, CFBDAuthError, CFBDError, ENDPOINTS
from .sources.market import load_markets

log = logging.getLogger("cfbmeta")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -- scheduling helpers ------------------------------------------------------
def local_now(config: Config) -> dt.datetime:
    return dt.datetime.now(ZoneInfo(config.timezone))


def should_run_now(config: Config, now: Optional[dt.datetime] = None) -> bool:
    """True when the local hour matches the configured send hour.

    GitHub Actions cron only speaks UTC, so the workflow fires at both candidate
    UTC hours and this gate decides which one is actually 7am locally. That way
    the email lands at 7am through the daylight-saving change in November
    without touching the schedule.
    """
    now = now or local_now(config)
    return now.hour == config.send_hour_local


def resolve_week(client, config: Config, season: int, now: Optional[dt.datetime] = None) -> Optional[int]:
    """The week whose games are still ahead of us — i.e. this weekend's slate."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)

    try:
        calendar = client.calendar(season)
    except Exception as exc:  # noqa: BLE001
        log.warning("calendar unavailable (%s); inferring week from the schedule", exc)
        calendar = []

    candidates = []
    for entry in calendar:
        from .sources.cfbd import pick, pick_float

        week = pick_float(entry, "week")
        last = parse_start(pick(entry, "lastGameStart", "last_game_start", "endDate", "end_date"))
        if week is None or last is None:
            continue
        if last >= now:
            candidates.append(int(week))
    if candidates:
        return min(candidates)

    # Fallback: earliest week that still has an unplayed game.
    try:
        from .sources.cfbd import pick, pick_float

        upcoming = []
        for game in client.games(season):
            start = parse_start(pick(game, "startDate", "start_date"))
            week = pick_float(game, "week")
            if start and week is not None and start >= now:
                upcoming.append(int(week))
        if upcoming:
            return min(upcoming)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not infer week from schedule: %s", exc)
    return None


def make_client(config: Config) -> CFBDClient:
    return CFBDClient(
        base_url=config.cfbd_base_url,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        cache_dir=config.cache_dir or None,
        cache_ttl_minutes=config.cache_ttl_minutes,
    )


# -- commands ----------------------------------------------------------------
def build_projections(client, config: Config, season: int, week: int):
    """Load every model input and project the slate."""
    log.info("building week %d of the %d season", week, season)

    book = build_rating_book(client, season, week=week)
    usable = book.usable_sources()
    if not usable:
        raise CFBDError(
            f"no usable rating source for {season}. Source audit:\n  "
            + "\n  ".join(book.provenance_lines())
        )
    log.info("ratings for %d teams from %s (season %d)", len(book), ", ".join(usable), season)

    hfa_model = build_hfa_model(
        client,
        seasons=range(season - 4, season + 1),
        league_hfa=config.league_hfa,
        shrink_games=config.hfa_shrink_games,
        hfa_min=config.hfa_min,
        hfa_max=config.hfa_max,
        altitude_bonus_per_1k_ft=config.altitude_bonus_per_1k_ft,
        altitude_threshold_ft=config.altitude_threshold_ft,
    )
    coach_model = build_coach_model(client, season, cap=config.coach_adj_cap)
    situational = build_situational_model(
        client,
        season,
        rest_day_value=config.rest_day_value,
        bye_week_bonus=config.bye_week_bonus,
        short_week_penalty=config.short_week_penalty,
        travel_penalty_per_1k_mi=config.travel_penalty_per_1k_mi,
        rest_cap=config.rest_adj_cap,
        travel_cap=config.travel_adj_cap,
    )

    games = client.games(season, week=week, season_type=config.season_type)
    if not games:
        log.warning("no games scheduled for week %d", week)
        return [], book

    markets = load_markets(client, season, week, config.season_type)
    projections = project_slate(
        games, book, config, hfa_model, coach_model, situational, markets
    )
    log.info("projected %d games", len(projections))
    return projections, book


def cmd_run(args, config: Config) -> int:
    now = local_now(config)
    if args.check_time and not should_run_now(config, now):
        log.info(
            "local time is %s; the send hour is %02d:00 %s. Exiting without sending.",
            now.strftime("%H:%M"), config.send_hour_local, config.timezone,
        )
        return 0

    season = args.season or config.resolved_season(now.date())
    client = make_client(config)

    # `is not None`, not truthiness: week 0 is a real week and would otherwise
    # be silently replaced by auto-detection.
    week = args.week if args.week is not None else resolve_week(client, config, season)
    if week is None:
        log.info("no upcoming week found for %d — the season is likely over.", season)
        return 0

    projections, book = build_projections(client, config, season, week)
    if not projections:
        log.info("nothing to report for week %d", week)
        return 0

    message = render(projections, config, week, season, generated_at=now, book=book)

    if args.out:
        Path(args.out).write_text(message["html"])
        log.info("wrote %s", args.out)

    if args.no_email:
        print(message["text"])
        return 0

    from .email_send import send_email

    sent = send_email(
        message["subject"], message["html"], message["text"], dry_run=args.dry_run
    )
    if not sent and args.dry_run:
        print(message["text"])
    return 0


def cmd_preview(args, config: Config) -> int:
    args.no_email = True
    args.check_time = False
    args.out = args.out or "preview.html"
    return cmd_run(args, config)


def cmd_doctor(args, config: Config) -> int:
    """Verify credentials and probe every endpoint the forecast depends on.

    Worth running once after setting the API key, and any time CFBD changes
    something. It reports the resolved path and the field names actually
    present, which is exactly what's needed to spot a renamed field.
    """
    season = args.season or config.resolved_season()
    print(f"cfbmeta doctor — season {season}\n" + "=" * 62)

    try:
        client = make_client(config)
    except CFBDAuthError as exc:
        print(f"FAIL  credentials: {exc}")
        return 1
    client.cache_dir = None  # always hit the network here
    print("OK    API key present")

    probes = [
        ("calendar", lambda: client.calendar(season)),
        ("games", lambda: client.games(season, week=1)),
        ("lines", lambda: client.lines(season, week=1)),
        ("teams", lambda: client.teams(season)),
        ("venues", lambda: client.venues()),
        ("sp", lambda: client.sp_ratings(season)),
        ("fpi", lambda: client.fpi_ratings(season)),
        ("srs", lambda: client.srs_ratings(season)),
        ("elo", lambda: client.elo_ratings(season)),
        ("talent", lambda: client.talent(season)),
        ("returning", lambda: client.returning_production(season)),
        ("coaches", lambda: client.coaches(season)),
    ]

    failures: List[str] = []
    for name, probe in probes:
        try:
            rows = probe()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name:<11} {type(exc).__name__}: {exc}")
            failures.append(name)
            continue

        path = client._resolved_paths.get(name, ENDPOINTS.get(name, ("?",))[0])
        count = len(rows)
        if not count:
            print(f"WARN  {name:<11} {path} returned 0 rows")
            continue
        fields = sorted(rows[0].keys())[:9] if isinstance(rows[0], dict) else []
        print(f"OK    {name:<11} {path} — {count} rows; fields: {', '.join(fields)}")
        if args.verbose and isinstance(rows[0], dict):
            print(f"      sample: {json.dumps(rows[0])[:400]}")

    print("-" * 62)
    if failures:
        print(f"{len(failures)} endpoint(s) failed: {', '.join(failures)}")
        print("A failure here usually means a tier restriction or a renamed path.")
        return 1

    # Which rating sources actually have data for this season? In August the
    # honest answer is "not all of them", and that must be visible.
    print("-" * 62)
    print(f"Rating source audit for season {season}:")
    audit_book = build_rating_book(client, season)
    for source in sorted(audit_book.provenance):
        status = audit_book.provenance[source]
        print(f"  {'OK  ' if status.usable else 'WARN'}  {status.describe()}")
    if not audit_book.usable_sources():
        print("  No usable source: the forecast cannot run for this season yet.")
        return 1

    # End-to-end: can we actually produce a slate?
    try:
        week = resolve_week(client, config, season)
        if week is None:
            week = 1
        projections, _ = build_projections(client, config, season, week)
        print(f"OK    end-to-end — projected {len(projections)} games for week {week}")
        for proj in projections[:3]:
            print(
                f"      {proj.matchup:<44} model {proj.projected_line:<22}"
                f" market {proj.market_spread if proj.market_spread is not None else 'n/a'}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  end-to-end: {type(exc).__name__}: {exc}")
        return 1

    print("\nAll checks passed.")
    return 0


def cmd_backtest(args, config: Config) -> int:
    season = args.season or config.resolved_season()
    seasons = list(range(season - args.years, season)) if not args.seasons else args.seasons
    client = make_client(config)

    print(f"Backtesting seasons {seasons} (basis: {args.basis})...")
    hfa_model = build_hfa_model(client, seasons=range(min(seasons) - 3, min(seasons)))
    coach_model = build_coach_model(client, min(seasons), cap=config.coach_adj_cap)

    rows = collect_rows(
        client, config, seasons, basis=args.basis, hfa_model=hfa_model, coach_model=coach_model
    )
    if not rows:
        print("No completed games collected; nothing to measure.")
        return 1

    weights = config.weights.normalized()
    weights.pop("market", None)
    fitted_shrink = None
    if args.fit:
        fitted = fit_weights(rows)
        if fitted:
            weights = fitted
        fitted_shrink = fit_edge_shrink(rows, weights)

    result = evaluate(rows, weights, basis=args.basis)
    print()
    print(result.summary())
    if fitted_shrink is not None:
        print(f"  edge shrink  {fitted_shrink:6.3f} (fraction of our disagreement "
              f"with the market that is real signal)")
        if args.basis == "same":
            print("               ignore this figure: same-season ratings inflate it.")

    if args.fit and weights and args.write:
        if args.basis == "same":
            print("\nRefusing to write config from a lookahead-contaminated backtest. "
                  "Re-run with --basis prior.")
            return 1
        for source, value in weights.items():
            if hasattr(config.weights, source):
                setattr(config.weights, source, round(value, 4))
        config.sigma_margin = round(result.sigma, 2)
        if fitted_shrink is not None:
            config.edge_shrink = fitted_shrink
        config.save()
        print(f"\nWrote fitted weights, sigma={config.sigma_margin} and "
              f"edge_shrink={config.edge_shrink} to config.yml")

    if args.fit_key_numbers:
        key_weights = fit_key_numbers(rows, sigma=result.sigma or config.sigma_margin)
        KEY_NUMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_NUMBERS_PATH.write_text(json.dumps(key_weights, indent=2, sort_keys=True))
        print(f"Wrote {len(key_weights)} key-number weights to {KEY_NUMBERS_PATH}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfbmeta", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.yml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build the slate and email it")
    run.add_argument("--week", type=int)
    run.add_argument("--season", type=int)
    run.add_argument("--out", help="also write the HTML here")
    run.add_argument("--no-email", action="store_true", help="print instead of sending")
    run.add_argument("--dry-run", action="store_true", help="render and log, but don't send")
    run.add_argument(
        "--check-time",
        action="store_true",
        help="exit unless the local hour matches send_hour_local (used by cron)",
    )
    run.set_defaults(func=cmd_run)

    preview = sub.add_parser("preview", help="render to a file without sending")
    preview.add_argument("--week", type=int)
    preview.add_argument("--season", type=int)
    preview.add_argument("--out", help="output path (default preview.html)")
    preview.set_defaults(func=cmd_preview, dry_run=False)

    doctor = sub.add_parser("doctor", help="verify credentials and every endpoint")
    doctor.add_argument("--season", type=int)
    doctor.set_defaults(func=cmd_doctor)

    back = sub.add_parser("backtest", help="measure accuracy and fit weights")
    back.add_argument("--season", type=int)
    back.add_argument("--years", type=int, default=4, help="how many prior seasons")
    back.add_argument("--seasons", type=int, nargs="+")
    back.add_argument(
        "--basis",
        choices=("prior", "same"),
        default="prior",
        help="'prior' uses last season's ratings (no lookahead); 'same' is "
             "contaminated and only useful for ranking sources against each other",
    )
    back.add_argument("--fit", action="store_true", help="refit blend weights")
    back.add_argument("--write", action="store_true", help="save fitted weights to config.yml")
    back.add_argument("--fit-key-numbers", action="store_true")
    back.set_defaults(func=cmd_backtest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    # Pick up a local .env before anything reads credentials from the env.
    load_dotenv()
    config = Config.load(args.config)

    try:
        return args.func(args, config)
    except CFBDAuthError as exc:
        log.error("%s", exc)
        return 1
    except CFBDError as exc:
        log.error("data error: %s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
