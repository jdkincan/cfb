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
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo

from .adjustments import build_situational_model, parse_start
from .backtest import (
    collect_rows, evaluate, fit_edge_shrink, fit_key_numbers, fit_weights, save_result,
)
from .coaching import build_coach_model
from .config import Config, load_dotenv, update_in_place
from .hfa import build_hfa_model
from .model import project_slate
from .probability import KEY_NUMBERS_PATH
from .ratings import build_rating_book, normalize_team
from .report import render
from .sources.cfbd import CFBDClient, CFBDAuthError, CFBDError, ENDPOINTS
from .sources.cfbd import pick as pick_any
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


def make_client(config: Config, refresh: bool = False) -> CFBDClient:
    """Build the API client. ``refresh`` disables the local disk cache.

    The cache is a local-development convenience only: it lives under data/,
    which is gitignored, so a scheduled run in CI starts with none and always
    fetches live.
    """
    return CFBDClient(
        base_url=config.cfbd_base_url,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        cache_dir=None if refresh else (config.cache_dir or None),
        cache_ttl_minutes=config.cache_ttl_minutes,
    )


# -- commands ----------------------------------------------------------------
def build_projections(client, config: Config, season: int, week: int, as_of=None,
                      window_days: Optional[int] = None):
    """Load every model input and project the slate.

    ``as_of`` anchors the slate window. It exists because a CFBD week can
    contain two playing weekends, so 'which weekend' is a separate question
    from 'which week' — and previewing the later one needs a way to say so.
    """
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
        log.warning("no games scheduled for week %d of %d", week, season)
        return [], book, {}

    from .model import clip_to_slate_window

    window_days = config.slate_window_days if window_days is None else window_days

    games = clip_to_slate_window(games, window_days, now=as_of)

    fbs_teams = None
    if config.fbs_only:
        try:
            from .ratings import normalize_team as _norm
            from .sources.cfbd import pick as _pick

            fbs_teams = {
                _norm(_pick(t, "school", "team"))
                for t in client.fbs_teams(season)
                if _pick(t, "school", "team")
            }
            log.info("restricting slate to %d FBS teams", len(fbs_teams))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load the FBS team list: %s", exc)

    from .availability import load as load_availability
    from .weather import build_weather_model

    availability = load_availability(season, week)
    weather = build_weather_model(client, season, week)
    if not weather.available and weather.reason:
        log.info("%s", weather.reason)

    markets = load_markets(client, season, week, config.season_type)

    # A second odds feed, if one is configured. This widens line shopping well
    # past CFBD's three retail books and, where Pinnacle quoted the game,
    # swaps the grading benchmark to a price that actually moves on sharp
    # money. No key, no change.
    from .sources import oddsapi

    try:
        enriched = oddsapi.enrich(
            markets,
            regions=config.odds_api_regions,
            prefer_sharp_benchmark=config.prefer_sharp_benchmark,
            timeout=config.request_timeout,
        )
        if enriched:
            log.info("odds api widened the book set for %d game(s)", enriched)
    except Exception as exc:  # noqa: BLE001 - a second feed must never break a run
        log.warning("odds api enrichment skipped: %s", exc)

    projections = project_slate(
        games, book, config, hfa_model, coach_model, situational, markets,
        fbs_teams, availability, weather,
    )
    log.info("projected %d games", len(projections))

    # Kelly sized each bet alone; scale the card for correlation and exposure.
    from .portfolio import apply as apply_portfolio, positions_from_projections

    positions = positions_from_projections(projections)
    portfolio = apply_portfolio(
        positions,
        max_weekly_units=config.max_weekly_units,
        within_group_rho=config.within_group_correlation,
        max_units_per_play=config.max_units_per_play,
    )
    scaled = {p.key: p.units for p in portfolio.positions}
    for proj in projections:
        if proj.spread_bet and str(proj.game_id) in scaled:
            proj.spread_bet.units = scaled[str(proj.game_id)]

    extras = {"markets": markets, "coach_model": coach_model,
              "portfolio": portfolio, "availability": availability,
              "weather": weather}

    if config.verify_spplus:
        extras["spplus_check"] = _verify_spplus(client, config, season)

    if config.spotlight_team:
        from .spotlight import build_spotlight

        try:
            from .model import detect_defense_sign, project_game

            defense_sign = detect_defense_sign(book)

            def project_one(game):
                gid = pick_any(game, "id", "gameId")
                return project_game(
                    game, book, config, hfa_model, coach_model, situational,
                    markets.get(gid), defense_sign,
                )

            extras["spotlight"] = build_spotlight(
                projections, config.spotlight_team, client=client, season=season,
                book=book, coach_model=coach_model,
                player_limit=config.spotlight_player_count,
                all_games=games, project_one=project_one,
            )
        except Exception as exc:  # noqa: BLE001 - never break a run over a bonus section
            log.warning("could not build the %s spotlight: %s", config.spotlight_team, exc)

    if config.archive_runs:
        from .archive import snapshot

        try:
            snapshot(season, week, book=book, projections=projections,
                     markets=markets, config=config)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not archive this run: %s", exc)

    return projections, book, extras


def _verify_spplus(client, config: Config, season: int):
    """Check CFBD's SP+ against the ESPN article of record."""
    from .ratings import normalize_team
    from .sources.cfbd import pick, pick_float
    from .sources.spplus_check import check

    try:
        def grab(year):
            return {
                normalize_team(pick(r, "team")): pick_float(r, "rating")
                for r in client.sp_ratings(year)
                if pick(r, "team") and pick_float(r, "rating") is not None
            }

        return check(config.spplus_article_id, grab(season), grab(season - 1))
    except Exception as exc:  # noqa: BLE001
        log.warning("SP+ publication check failed: %s", exc)
        return None


def cmd_run(args, config: Config) -> int:
    now = local_now(config)
    if args.check_time and not should_run_now(config, now):
        log.info(
            "local time is %s; the send hour is %02d:00 %s. Exiting without sending.",
            now.strftime("%H:%M"), config.send_hour_local, config.timezone,
        )
        return 0

    season = args.season or config.resolved_season(now.date())
    client = make_client(config, refresh=getattr(args, 'refresh', False))

    # `is not None`, not truthiness: week 0 is a real week and would otherwise
    # be silently replaced by auto-detection.
    as_of = parse_start(getattr(args, "as_of", None))
    week = args.week if args.week is not None else resolve_week(
        client, config, season, now=as_of
    )
    if week is None:
        log.info("no upcoming week found for %d — the season is likely over.", season)
        return 0

    projections, book, extras = build_projections(
        client, config, season, week, as_of=as_of,
        window_days=getattr(args, 'window', None),
    )
    if not projections:
        log.info("nothing to report for week %d", week)
        return 0

    message = render(projections, config, week, season, generated_at=now,
                     book=book, extras=extras)

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


def cmd_trend(args, config: Config) -> int:
    """Weekly opponent-adjusted efficiency, as a time series.

    CFBD's SP+, FPI and SRS all silently ignore the week parameter — they are
    season-final numbers, and no weekly history exists to fetch. This rebuilds
    an SP+-shaped rating from per-game advanced stats, refit from exactly the
    games completed before each week, which makes the series available for any
    season including ones long finished.
    """
    from .sources.efficiency import team_trend, weekly_series
    from .sources.cfbd import pick

    season = args.season or config.resolved_season()
    client = make_client(config, refresh=args.refresh)

    rows = client.advanced_game_stats(season, season_type=config.season_type)
    if not rows:
        print(f"no per-game advanced stats for {season}.")
        return 1
    games = client.games(season, season_type=config.season_type)

    fbs = None
    if config.fbs_only:
        try:
            fbs = {
                normalize_team(pick(t, "school", "team"))
                for t in client.fbs_teams(season)
                if pick(t, "school", "team")
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load the FBS list: %s", exc)

    weeks = sorted({int(w) for w in (pick(r, "week") for r in rows) if w is not None})
    if not weeks:
        print("no weeks found in the advanced stats.")
        return 1
    # Fit before each week that has games, plus one past the end so the final
    # state of the season is included.
    series = weekly_series(
        rows, games, [w for w in weeks] + [max(weeks) + 1],
        season=season, center_on=fbs,
    )
    if not series:
        print("not enough completed games to fit a rating yet.")
        return 1

    if args.write:
        from .archive import write_efficiency_series

        path = write_efficiency_series(season, series)
        print(f"wrote {path}")

    latest = series[max(series)]
    if args.team:
        trend = team_trend(series, args.team)
        if not trend:
            print(f"no rating history for {args.team!r} in {season}.")
            return 1
        print(f"{args.team} — weekly efficiency rating, {season}")
        print(f"{'week':>5}  {'rating':>7}  {'rank':>5}  {'move':>6}")
        previous = None
        for week, value, rank in trend:
            move = "" if previous is None else f"{value - previous:+.1f}"
            print(f"{week:>5}  {value:>7.1f}  {rank:>5}  {move:>6}")
            previous = value
        return 0

    top = latest.ranked()[: args.top]
    prior = series.get(sorted(series)[-2]) if len(series) > 1 else None
    print(f"Efficiency rating through week {latest.week - 1} of {season} "
          f"— fit on {latest.observations} team-games across {len(latest)} teams, "
          f"ranked among {len(latest.ranked())}")
    print(f"{'#':>3}  {'team':<24} {'rating':>7}  {'1wk':>6}")
    for i, (team, value) in enumerate(top, 1):
        move = ""
        if prior is not None and team in prior.points:
            move = f"{value - prior.points[team]:+.1f}"
        print(f"{i:>3}  {team:<24} {value:>7.1f}  {move:>6}")
    return 0


def cmd_simulate(args, config: Config) -> int:
    """Monte Carlo the season from the same numbers the weekly readout uses.

    Base margins come from project_game, so the simulation and the Thursday
    email cannot disagree about how good a team is.
    """
    from .coaching import build_coach_model
    from .hfa import build_hfa_model
    from .model import detect_defense_sign, project_game
    from .simulate import actual_win_totals, coverage_report, simulate
    from .adjustments import build_situational_model
    from .sources.cfbd import pick

    season = args.season or config.resolved_season()
    client = make_client(config, refresh=args.refresh)

    games = client.games(season, season_type=config.season_type)
    if not games:
        print(f"no schedule for {season}.")
        return 1

    finished = sum(
        1 for g in games
        if pick(g, "homePoints", "home_points") is not None
    )
    rating_season = args.rating_season or season
    if args.preseason and finished and rating_season == season:
        # Replaying a finished season from its own final ratings is lookahead:
        # those ratings already know how every game turned out. Fall back to
        # what a real August forecast would have had.
        rating_season = season - 1
        log.info(
            "preseason replay of a completed season: using %d ratings, not %d, "
            "to avoid lookahead", rating_season, season,
        )

    book = build_rating_book(client, rating_season, week=args.through)
    if not book.usable_sources():
        print(f"no usable rating source for {rating_season}; cannot simulate.")
        return 1

    hfa_model = build_hfa_model(
        client, seasons=range(rating_season - 4, rating_season + 1),
        league_hfa=config.league_hfa, shrink_games=config.hfa_shrink_games,
        hfa_min=config.hfa_min, hfa_max=config.hfa_max,
        altitude_bonus_per_1k_ft=config.altitude_bonus_per_1k_ft,
        altitude_threshold_ft=config.altitude_threshold_ft,
    )
    coach_model = build_coach_model(client, rating_season, cap=config.coach_adj_cap)
    situational = build_situational_model(
        client, season, rest_day_value=config.rest_day_value,
        bye_week_bonus=config.bye_week_bonus,
        short_week_penalty=config.short_week_penalty,
        travel_penalty_per_1k_mi=config.travel_penalty_per_1k_mi,
        rest_cap=config.rest_adj_cap, travel_cap=config.travel_adj_cap,
    )

    fbs = None
    if config.fbs_only:
        try:
            fbs = {
                normalize_team(pick(t, "school", "team"))
                for t in client.fbs_teams(season)
                if pick(t, "school", "team")
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load the FBS list: %s", exc)

    defense_sign = detect_defense_sign(book)

    def base_margin(game):
        proj = project_game(
            game, book, config, hfa_model, coach_model, situational,
            None, defense_sign,
        )
        # projected_margin defaults to 0.0, and project_game returns early
        # without setting it when no source rates both teams. Passing that 0.0
        # through would turn "we cannot rate this FCS opponent" into "pick'em",
        # which is how a 95% win becomes a coin flip.
        if proj is None or not proj.components:
            return None
        return proj.projected_margin

    blind = games
    if args.preseason:
        # Hide results so the whole season is projected, not just what is left.
        # This is how the simulator is validated, and how a "what did we think
        # in August" run is produced after the fact.
        drop = ("homePoints", "awayPoints", "home_points", "away_points")
        blind = [{k: v for k, v in g.items() if k not in drop} for g in games]

    sim = simulate(
        blind, base_margin, season=season, sims=args.sims,
        sigma_game=config.sigma_margin, sigma_team=config.sigma_team,
        eligible=fbs, seed=args.seed,
    )
    if not sim.teams:
        print("nothing simulable: no game could be projected.")
        return 1

    print(f"{season} regular season — {sim.sims:,} simulations")
    detail = f"  {sim.games_simulated} games projected, {sim.games_locked} already final"
    if sim.games_base_rate:
        detail += (f" ({sim.games_base_rate} vs unrated opponents at the "
                   f"FBS-vs-FCS base rate)")
    print(detail)
    if rating_season != season:
        print(f"  rated from {rating_season} final ratings (no lookahead)")
    print(f"  sigma: {sim.sigma_game:.1f} per game, {sim.sigma_team:.1f} per team-season")

    truth = actual_win_totals(games, eligible=fbs)
    if truth and sim.games_simulated > 0.5 * (sim.games_simulated + sim.games_locked):
        report = coverage_report(sim, truth, 0.80)
        if report["n"]:
            print(f"  vs what actually happened: mean-win MAE {report['mae']:.2f}, "
                  f"80% interval covered {report['coverage']:.1%} of {report['n']} teams")

    if args.team:
        key = normalize_team(args.team)
        outcome = sim.teams.get(key)
        if outcome is None:
            print(f"\n{args.team!r} is not in the simulated field.")
            return 1
        low, high = outcome.interval(0.80)
        conf = outcome.conference or "Independent"
        print(f"\n{outcome.display} ({conf}) — {outcome.scheduled} games")
        print(f"  projected {outcome.mean_wins:.1f} wins   80% interval {low}-{high}")
        if outcome.played:
            print(f"  {outcome.actual_wins}-{outcome.played - outcome.actual_wins} so far")
        if outcome.conference_title is not None:
            print(f"  conference title {outcome.conference_title:.1%}")
        print(f"\n  {'wins':>5}  {'chance':>7}  {'at least':>9}")
        for wins in sorted(outcome.win_counts):
            share = outcome.win_counts[wins]
            if share < 0.001:
                continue
            bar = "#" * max(1, round(share * 60))
            print(f"  {wins:>5}  {share:>6.1%}  {outcome.probability_of_at_least(wins):>8.1%}  {bar}")
        return 0

    if args.conference:
        rows = sim.conference(args.conference)
        if not rows:
            print(f"\nno teams found in conference {args.conference!r}.")
            return 1
        print(f"\n{args.conference}")
        print(f"  {'team':<22} {'wins':>5} {'80% int':>9} {'conf':>5} {'title':>7}")
        for outcome in rows:
            low, high = outcome.interval(0.80)
            title = ("     -" if outcome.conference_title is None
                     else f"{outcome.conference_title:>6.1%}")
            print(f"  {outcome.display:<22} {outcome.mean_wins:>5.1f} "
                  f"{f'{low}-{high}':>9} {outcome.mean_conference_wins:>5.1f} "
                  f"{title}")
        return 0

    print(f"\n  {'#':>3} {'team':<22} {'wins':>5} {'80% int':>9} {'conf':<18} {'title':>7}")
    for i, outcome in enumerate(sim.ranked(limit=args.top), 1):
        low, high = outcome.interval(0.80)
        title = ("     -" if outcome.conference_title is None
                 else f"{outcome.conference_title:>6.1%}")
        conf = outcome.conference or "Independent"
        print(f"  {i:>3} {outcome.display:<22} {outcome.mean_wins:>5.1f} "
              f"{f'{low}-{high}':>9} {conf[:18]:<18} {title}")

    titles = sorted(
        (t for t in sim.teams.values() if t.conference_title is not None),
        key=lambda t: -t.conference_title,
    )
    if titles:
        print(f"\n  conference favourites")
        seen = set()
        for outcome in titles:
            if outcome.conference in seen:
                continue
            seen.add(outcome.conference)
            print(f"    {outcome.conference:<24} {outcome.display:<20} "
                  f"{outcome.conference_title:>6.1%}")
        print("    (most conference wins, random tiebreak — no championship "
              "games or head-to-head rules)")
    return 0


def _season_simulation(client, config: Config, season: int, sims: int,
                       rating_season: Optional[int] = None, blind: bool = False):
    """Run the full-season simulation the win-total card is priced off."""
    from .adjustments import build_situational_model
    from .coaching import build_coach_model
    from .hfa import build_hfa_model
    from .model import detect_defense_sign, project_game
    from .simulate import simulate
    from .sources.cfbd import pick as _pick

    rs = rating_season if rating_season is not None else season
    book = build_rating_book(client, rs)
    games = client.games(season, season_type=config.season_type)
    hfa_model = build_hfa_model(
        client, seasons=range(rs - 4, rs + 1), league_hfa=config.league_hfa,
        shrink_games=config.hfa_shrink_games, hfa_min=config.hfa_min,
        hfa_max=config.hfa_max,
        altitude_bonus_per_1k_ft=config.altitude_bonus_per_1k_ft,
        altitude_threshold_ft=config.altitude_threshold_ft)
    coach_model = build_coach_model(client, rs, cap=config.coach_adj_cap)
    situational = build_situational_model(
        client, season, rest_day_value=config.rest_day_value,
        bye_week_bonus=config.bye_week_bonus,
        short_week_penalty=config.short_week_penalty,
        travel_penalty_per_1k_mi=config.travel_penalty_per_1k_mi,
        rest_cap=config.rest_adj_cap, travel_cap=config.travel_adj_cap)
    fbs = {normalize_team(_pick(t, "school", "team"))
           for t in client.fbs_teams(season) if _pick(t, "school", "team")}
    sign = detect_defense_sign(book)

    def margin(game):
        proj = project_game(game, book, config, hfa_model, coach_model,
                            situational, None, sign)
        return proj.projected_margin if (proj and proj.components) else None

    feed = games
    if blind:
        drop = ("homePoints", "awayPoints", "home_points", "away_points")
        feed = [{k: v for k, v in g.items() if k not in drop} for g in games]
    sim = simulate(feed, margin, season=season, sims=sims,
                   sigma_game=config.sigma_margin, sigma_team=config.sigma_team,
                   eligible=fbs, seed=17)
    return sim, games, fbs


def cmd_wintotals(args, config: Config) -> int:
    """Price posted season win totals against the simulation."""
    from .wintotals import build_card, load_lines

    season = args.season or config.resolved_season()
    lines = load_lines(args.lines)
    manual = {l.key for l in lines}

    if args.scrape:
        from .sources.wintotal_scrape import load as scrape_totals

        client_for_names = make_client(config, refresh=args.refresh)
        from .sources.cfbd import pick as _pick

        known = {normalize_team(_pick(t, "school", "team"))
                 for t in client_for_names.fbs_teams(season)
                 if _pick(t, "school", "team")}
        scraped = scrape_totals(known_teams=known)
        # Anything typed by hand wins: a real quote beats a scraped one.
        added = [l for l in scraped if l.key not in manual]
        lines = list(lines) + added
        print(f"scraped {len(scraped)} posted totals, kept {len(added)} "
              f"({len(manual)} overridden by win-totals.yml)")

    if not lines:
        print("No posted totals found. Try --scrape, or add them to win-totals.yml:\n\n"
              "  totals:\n    Arkansas: 4.5\n    Georgia:\n      total: 10.5\n"
              "      over: -140\n      under: 115\n")
        return 1

    client = make_client(config, refresh=args.refresh)
    sim, _, _ = _season_simulation(client, config, season, args.sims)
    card = build_card(
        sim, lines, shrink=config.wintotal_shrink,
        min_edge_wins=config.wintotal_min_edge_wins,
        kelly_multiplier=config.kelly_fraction,
        bankroll_units=config.bankroll_units,
        max_units=config.wintotal_max_units,
        max_total_units=config.wintotal_max_total_units,
        exclude=config.bet_exclusions,
    )
    plays = [b for b in card if b.is_play]

    print(f"\nWIN TOTALS — {season}  ({len(lines)} posted, {sim.sims:,} simulated seasons)")
    print("UNVALIDATED: no history of posted totals exists to test this against.")
    print(f"shrink {config.wintotal_shrink:.2f}, threshold "
          f"{config.wintotal_min_edge_wins:.2f} wins\n")
    if not plays:
        print("No plays clear the threshold.")
    else:
        print(f"{'team':<20}{'side':>6}{'line':>7}{'price':>7}{'model':>7}"
              f"{'edge':>7}{'win%':>7}{'EV':>7}{'units':>7}")
        for b in plays:
            print(f"{b.display:<20}{b.side:>6}{b.total:>7.1f}{b.price:>7}"
                  f"{b.model_wins:>7.1f}{b.edge_wins:>+7.2f}"
                  f"{100*b.win_probability:>6.1f}%{100*b.expected_value:>+6.1f}%"
                  f"{b.units:>7.2f}")
        print(f"\n{len(plays)} plays, {sum(b.units for b in plays):.2f} units at risk")

    if args.all:
        print("\nPASSED:")
        for b in card:
            if b.is_play:
                continue
            note = b.notes[0] if b.notes else ""
            print(f"  {b.display:<20}{b.total:>6.1f}  model {b.model_wins:>5.1f}  {note}")
    return 0


def cmd_doctor(args, config: Config) -> int:
    """Verify credentials and probe every endpoint the forecast depends on.

    Worth running once after setting the API key, and any time CFBD changes
    something. It reports the resolved path and the field names actually
    present, which is exactly what's needed to spot a renamed field.
    """
    season = args.season or config.resolved_season()
    print(f"cfbmeta doctor — season {season}\n" + "=" * 62)

    try:
        client = make_client(config, refresh=True)
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

    # The second odds feed is optional, so its absence is not a failure — but
    # running without it silently would hide that edge is being measured
    # against three retail books.
    print("-" * 62)
    if os.getenv("ODDS_API_KEY"):
        from .sources import oddsapi

        try:
            odds = oddsapi.fetch_odds(regions=config.odds_api_regions,
                                      timeout=config.request_timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN  odds api    fetch failed: {type(exc).__name__}: {exc}")
        else:
            if not odds:
                print("WARN  odds api    key set but no games returned")
            else:
                books = sorted({q.book for g in odds for q in g.quotes})
                sharp = [g for g in odds if g.sharp_spread() is not None]
                print(f"OK    odds api    {len(odds)} games, {len(books)} books: "
                      f"{', '.join(books[:10])}")
                print(f"      sharp benchmark available for {len(sharp)}/{len(odds)} games")
    else:
        print("SKIP  odds api    ODDS_API_KEY not set — edge is graded against "
              "CFBD's retail books")

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
        projections, _, _ = build_projections(client, config, season, week)
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
    if args.basis == "reconstructed":
        # Two reconstructed components: ridge margin ratings and the in-season
        # efficiency rating, both refit from only the games completed before
        # each week. Measured on 2025 the 0.4/0.6 split beat either alone
        # (12.44 MAE against 12.91 and 13.81); --fit re-estimates it.
        weights = {"inseason": 0.60, "efficiency": 0.40}
    fitted_shrink = None
    if args.fit:
        fitted = fit_weights(rows)
        if fitted:
            weights = fitted
        fitted_shrink = fit_edge_shrink(rows, weights)

    result = evaluate(rows, weights, basis=args.basis)
    save_result(result, seasons, fitted_shrink)
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
        updates = {"sigma_margin": round(result.sigma, 2)}
        # An edge-shrink fitted on prior-season ratings measures what stale
        # numbers are worth against the close, not what the live model is
        # worth. Adopting it would silently set the operating value from the
        # wrong experiment — in either direction — so it is recorded and not
        # written. Only a point-in-time (snapshot) basis can settle it.
        if args.basis in ("snapshot", "reconstructed") and fitted_shrink is not None:
            updates["edge_shrink"] = fitted_shrink
        elif fitted_shrink is not None:
            print(
                f"\nNot writing edge_shrink={fitted_shrink}: a '{args.basis}' basis "
                "measures stale ratings against the closing line, which is not the "
                "quantity the live model uses. Recorded in calibration/backtest.json."
            )
        changed = update_in_place(
            updates, weights={k: round(v, 4) for k, v in weights.items()}
        )
        print("\nUpdated config.yml (comments preserved):")
        for line in changed:
            print(f"  {line}")

    if args.fit_key_numbers:
        key_weights = fit_key_numbers(rows, sigma=result.sigma or config.sigma_margin)
        KEY_NUMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_NUMBERS_PATH.write_text(json.dumps(key_weights, indent=2, sort_keys=True))
        print(f"Wrote {len(key_weights)} key-number weights to {KEY_NUMBERS_PATH}")

    return 0


def cmd_ledger(args, config: Config) -> int:
    """Show the bet ledger, or settle open bets from final scores."""
    from .ledger import summarize, settle
    from .sources.cfbd import pick, pick_float

    if args.settle:
        season = args.season or config.resolved_season()
        client = make_client(config)
        results, closing = {}, {}
        for week in range(1, 20):
            try:
                games = client.games(season, week=week)
            except Exception:  # noqa: BLE001
                break
            for game in games:
                hp = pick_float(game, "homePoints", "home_points")
                ap = pick_float(game, "awayPoints", "away_points")
                gid = pick(game, "id")
                if gid is not None and hp is not None and ap is not None:
                    results[str(gid)] = hp - ap
            try:
                for row in client.lines(season, week=week):
                    gid = pick(row, "id")
                    from .sources.market import consensus_line

                    line = consensus_line(row)
                    if gid is not None and line and line.get("spread") is not None:
                        closing[str(gid)] = line["spread"]
            except Exception:  # noqa: BLE001
                pass
        settled = settle(results, closing)
        print(f"Settled {settled} bet(s).\n")

    print(summarize().describe())
    return 0


def cmd_resize(args, config: Config) -> int:
    """Re-size a card down to the plays you actually intend to bet."""
    import csv as _csv

    from .archive import week_dir
    from .portfolio import Position, resize

    season = args.season or config.resolved_season()
    path = week_dir(season, args.week) / "projections.csv"
    if not path.exists():
        print(f"No archived slate at {path}. Run the week first.")
        return 1

    with path.open() as handle:
        rows = [r for r in _csv.DictReader(handle) if (r.get("bet_side") or "none") != "none"]

    positions = []
    for row in rows:
        try:
            units = float(row.get("units") or 0)
        except ValueError:
            continue
        if units <= 0:
            continue
        side_team = row["home_team"] if row["bet_side"] == "home" else row["away_team"]
        positions.append(Position(
            key=str(row["game_id"]),
            label=f"{side_team} {float(row['bet_line']):+.1f}",
            units=units,
            kickoff_bucket=(row.get("kickoff") or "")[:10],
            edge=float(row.get("edge") or 0),
        ))

    if not args.keep:
        print(f"Plays archived for week {args.week}:\n")
        for p in sorted(positions, key=lambda p: -abs(p.edge)):
            print(f"  {p.key:<12}{p.label:<28}{p.units:5.2f}u   edge {p.edge:+.1f}")
        print("\nRe-run with --keep <game_id> ... to size a chosen subset.")
        return 0

    result = resize(
        positions, args.keep,
        max_weekly_units=config.max_weekly_units,
        max_units_per_play=config.max_units_per_play,
        within_group_rho=config.within_group_correlation,
    )
    print(result.describe() + "\n")
    for p in result.positions:
        print(f"  {p.label:<28}{p.units:5.2f}u")
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
    run.add_argument(
        "--refresh", action="store_true",
        help="bypass the local cache and refetch everything",
    )
    run.add_argument(
        "--window", type=int,
        help="days of games to include from the anchor kickoff "
             "(default 6 = one weekend; use 14 for a whole CFBD week)",
    )
    run.add_argument(
        "--as-of", dest="as_of",
        help="anchor the slate window at this date (YYYY-MM-DD) instead of today; "
             "use it to preview a later weekend inside the same CFBD week",
    )
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
    preview.add_argument(
        "--refresh", action="store_true",
        help="bypass the local cache and refetch everything",
    )
    preview.add_argument(
        "--window", type=int,
        help="days of games to include from the anchor kickoff "
             "(default 6 = one weekend; use 14 for a whole CFBD week)",
    )
    preview.add_argument(
        "--as-of", dest="as_of",
        help="anchor the slate window at this date (YYYY-MM-DD) instead of today; "
             "use it to preview a later weekend inside the same CFBD week",
    )
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
        choices=("prior", "same", "reconstructed", "snapshot"),
        default="prior",
        help="'prior' uses last season's ratings (no lookahead); 'same' is "
             "contaminated and only useful for ranking sources against each other",
    )
    back.add_argument("--fit", action="store_true", help="refit blend weights")
    back.add_argument("--write", action="store_true", help="save fitted weights to config.yml")
    back.add_argument("--fit-key-numbers", action="store_true")
    back.set_defaults(func=cmd_backtest)

    led = sub.add_parser("ledger", help="show or settle the bet ledger")
    led.add_argument("--settle", action="store_true",
                     help="grade open bets from final scores and record CLV")
    led.add_argument("--season", type=int)
    led.set_defaults(func=cmd_ledger)

    wt = sub.add_parser(
        "wintotals", help="price posted season win totals against the simulation")
    wt.add_argument("--season", type=int)
    wt.add_argument("--sims", type=int, default=20000)
    wt.add_argument("--lines", help="path to a win totals file (default win-totals.yml)")
    wt.add_argument("--all", action="store_true", help="also show what was passed")
    wt.add_argument("--scrape", action="store_true",
                    help="pull posted totals from an odds aggregator; anything "
                         "in win-totals.yml overrides what is scraped")
    wt.add_argument("--refresh", action="store_true")
    wt.set_defaults(func=cmd_wintotals)

    tr = sub.add_parser(
        "trend", help="weekly efficiency ratings over time (the SP+ analog)")
    tr.add_argument("--season", type=int)
    tr.add_argument("--team", help="show one team's week-by-week history")
    tr.add_argument("--top", type=int, default=25, help="how many teams to list")
    tr.add_argument("--refresh", action="store_true",
                    help="bypass the local cache and refetch everything")
    tr.add_argument("--write", action="store_true",
                    help="also save the full series to archive/<season>/")
    tr.set_defaults(func=cmd_trend)

    simu = sub.add_parser("simulate", help="Monte Carlo the season: win totals and races")
    simu.add_argument("--season", type=int)
    simu.add_argument("--sims", type=int, default=20000)
    simu.add_argument("--team", help="full win distribution for one team")
    simu.add_argument("--conference", help="standings for one conference")
    simu.add_argument("--top", type=int, default=25)
    simu.add_argument("--rating-season", type=int, default=None,
                      help="which season's ratings to use (default: the target "
                           "season; a preseason replay drops to the prior one)")
    simu.add_argument("--through", type=int, default=None,
                      help="rate teams as of this week (default: current ratings)")
    simu.add_argument("--preseason", action="store_true",
                      help="hide results and project the whole season from scratch")
    simu.add_argument("--seed", type=int, default=0)
    simu.add_argument("--refresh", action="store_true",
                      help="bypass the local cache and refetch everything")
    simu.set_defaults(func=cmd_simulate)

    res = sub.add_parser("resize", help="re-size a card down to chosen plays")
    res.add_argument("--week", type=int, required=True)
    res.add_argument("--season", type=int)
    res.add_argument("--keep", nargs="*", default=[],
                     help="game ids to keep; omit to list what is available")
    res.set_defaults(func=cmd_resize)

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
