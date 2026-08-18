import datetime as dt
import os
from unittest import mock

import pytest

from cfbmeta.adjustments import build_situational_model
from cfbmeta.cli import resolve_week, should_run_now
from cfbmeta.coaching import build_coach_model
from cfbmeta.config import Config
from cfbmeta.email_send import (
    EmailConfigError,
    SMTPSettings,
    build_message,
    send_email,
)
from cfbmeta.hfa import build_hfa_model
from cfbmeta.model import project_slate
from cfbmeta.report import build_context, format_spread, render, render_html, render_text
from cfbmeta.sources.market import build_market_map

from conftest import SEASON, FakeClient


@pytest.fixture
def projections(client, book, config):
    games = [g for g in client.games(SEASON) if g["week"] == 6]
    return project_slate(
        games,
        book,
        config,
        build_hfa_model(client, [SEASON - 1, SEASON]),
        build_coach_model(client, SEASON, lookback_years=4),
        build_situational_model(client, SEASON),
        build_market_map(client.lines(SEASON, week=6)),
    )


@pytest.fixture
def context(projections, config):
    return build_context(projections, config, week=6, season=SEASON)


class TestFormatting:
    def test_spread_is_quoted_from_the_favorite(self):
        assert format_spread(7.0, "Texas") == "Texas -7.0"
        assert format_spread(-7.0, "Texas") == "Texas -7.0"

    def test_missing_spread_renders_a_dash(self):
        assert format_spread(None, "Texas") == "—"


class TestContext:
    def test_every_game_appears(self, context, projections):
        assert len(context["games"]) == len(projections)

    def test_plays_are_a_subset_of_games(self, context):
        assert len(context["plays"]) <= len(context["games"])
        assert all(p["is_play"] for p in context["plays"])

    def test_summary_tiles_are_populated(self, context):
        labels = [label for label, _ in context["summary_tiles"]]
        assert labels == ["Games", "Plays", "Units", "Avg edge"]

    def test_ledger_explains_the_projected_margin(self, context):
        game = context["games"][0]
        assert any("Blended ratings" in label for label, _ in game["ledger"])

    def test_neutral_site_ledger_says_so(self, context):
        neutral = next(g for g in context["games"] if g["neutral_site"])
        assert any("neutral" in str(value).lower() for _, value in neutral["ledger"])

    def test_altitude_is_called_out_in_the_ledger(self, context):
        wyoming = next(g for g in context["games"] if g["home_team"] == "Wyoming")
        assert any("altitude" in label.lower() for label, _ in wyoming["ledger"])

    def test_play_side_names_a_real_team(self, context):
        for play in context["plays"]:
            bet = play["spread_bet"]
            assert bet["side_team"] in (play["home_team"], play["away_team"])


class TestHTML:
    def test_renders_without_error(self, context):
        html = render_html(context)
        assert len(html) > 2000

    def test_contains_the_matchups(self, context):
        html = render_html(context)
        for game in context["games"][:3]:
            assert game["home_team"] in html

    def test_no_unrendered_jinja_left(self, context):
        html = render_html(context)
        assert "{{" not in html and "{%" not in html

    def test_uses_inline_styles_for_mail_clients(self, context):
        html = render_html(context)
        assert "<style" not in html.lower()
        assert 'style="' in html

    def test_apostrophes_are_escaped_not_mangled(self, context):
        html = render_html(context)
        # Hawai'i must survive autoescaping in a readable form.
        assert "Hawai" in html

    def test_empty_slate_still_renders(self, config):
        context = build_context([], config, week=6, season=SEASON)
        html = render_html(context)
        assert "No game cleared" in html

    def test_no_plays_shows_the_fallback_notice(self, client, book, config):
        # The threshold is applied when the bet is evaluated, so it has to be
        # set before projecting, not after.
        config.min_edge_points = 99.0
        games = [g for g in client.games(SEASON) if g["week"] == 6]
        quiet = project_slate(
            games, book, config, markets=build_market_map(client.lines(SEASON, week=6))
        )
        rebuilt = build_context(quiet, config, week=6, season=SEASON)
        assert rebuilt["plays"] == []
        assert "No game cleared" in render_html(rebuilt)


class TestText:
    def test_plain_text_alternative_renders(self, context):
        text = render_text(context)
        assert "FULL SLATE" in text
        assert "Sources:" in text

    def test_columns_do_not_collide(self, context):
        """Adjacent columns in the slate table must keep whitespace between them."""
        table = render_text(context).split("FULL SLATE")[1]
        rows = [ln for ln in table.splitlines() if " at " in ln or " vs " in ln]
        assert len(rows) == len(context["games"])
        for line in rows:
            # Model and market columns must not run together, as in
            # "Notre Dame -6.1Notre Dame -10.5".
            assert "  " in line.strip(), line
            fields = [f for f in line.split("  ") if f.strip()]
            assert len(fields) == 4, line

    def test_message_bundle_has_all_parts(self, projections, config):
        message = render(projections, config, week=6, season=SEASON)
        assert set(message) == {"subject", "html", "text"}
        assert "Week 6" in message["subject"]


class TestSchedulingGate:
    def test_fires_only_at_the_configured_hour(self):
        config = Config()
        config.timezone = "America/New_York"
        config.send_hour_local = 7
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        assert should_run_now(config, dt.datetime(2025, 10, 2, 7, 5, tzinfo=tz)) is True
        assert should_run_now(config, dt.datetime(2025, 10, 2, 8, 5, tzinfo=tz)) is False
        assert should_run_now(config, dt.datetime(2025, 10, 2, 6, 55, tzinfo=tz)) is False

    def test_gate_holds_across_the_daylight_saving_change(self):
        """11:00 and 12:00 UTC are both scheduled; exactly one is 7am local."""
        from zoneinfo import ZoneInfo

        config = Config()
        config.timezone = "America/New_York"
        config.send_hour_local = 7
        eastern = ZoneInfo("America/New_York")

        # October: EDT, so 11:00 UTC is 7am and 12:00 UTC is 8am.
        october = [
            dt.datetime(2025, 10, 2, h, tzinfo=dt.timezone.utc).astimezone(eastern)
            for h in (11, 12)
        ]
        assert [should_run_now(config, t) for t in october] == [True, False]

        # December: EST, so 12:00 UTC is 7am and 11:00 UTC is 6am.
        december = [
            dt.datetime(2025, 12, 4, h, tzinfo=dt.timezone.utc).astimezone(eastern)
            for h in (11, 12)
        ]
        assert [should_run_now(config, t) for t in december] == [False, True]

    def test_exactly_one_fire_per_thursday_all_season(self):
        from zoneinfo import ZoneInfo

        config = Config()
        config.timezone = "America/New_York"
        config.send_hour_local = 7
        eastern = ZoneInfo("America/New_York")

        thursday = dt.datetime(2025, 8, 28, tzinfo=dt.timezone.utc)
        for _ in range(20):  # the whole season, through the DST change
            fires = [
                should_run_now(
                    config, thursday.replace(hour=h).astimezone(eastern)
                )
                for h in (11, 12)
            ]
            assert sum(fires) == 1, f"{thursday.date()} fired {sum(fires)} times"
            thursday += dt.timedelta(days=7)


class TestWeekResolution:
    def test_picks_the_upcoming_week(self, client, config):
        # Fixture weeks start 2025-08-30 and run weekly.
        now = dt.datetime(SEASON, 10, 1, 12, tzinfo=dt.timezone.utc)
        assert resolve_week(client, config, SEASON, now) is not None

    def test_returns_none_after_the_season(self, client, config):
        now = dt.datetime(SEASON + 1, 3, 1, tzinfo=dt.timezone.utc)
        assert resolve_week(client, config, SEASON, now) is None

    def test_falls_back_to_the_schedule_without_a_calendar(self, config):
        client = FakeClient(fail={"calendar"})
        now = dt.datetime(SEASON, 10, 1, 12, tzinfo=dt.timezone.utc)
        assert resolve_week(client, config, SEASON, now) is not None


class TestEmailSettings:
    def test_requires_credentials(self):
        with mock.patch.dict(os.environ, {"SMTP_USER": "", "SMTP_PASSWORD": ""}, clear=True):
            with pytest.raises(EmailConfigError):
                SMTPSettings.from_env()

    def test_defaults_recipient_to_the_sender(self):
        env = {"SMTP_USER": "me@gmail.com", "SMTP_PASSWORD": "app-password"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SMTPSettings.from_env()
        assert settings.recipients == ["me@gmail.com"]
        assert settings.host == "smtp.gmail.com"
        assert settings.use_tls is True

    def test_parses_multiple_recipients(self):
        env = {
            "SMTP_USER": "me@gmail.com",
            "SMTP_PASSWORD": "x",
            "EMAIL_TO": "a@x.com, b@y.com; c@z.com",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SMTPSettings.from_env()
        assert settings.recipients == ["a@x.com", "b@y.com", "c@z.com"]

    def test_port_465_uses_implicit_tls(self):
        env = {"SMTP_USER": "me@x.com", "SMTP_PASSWORD": "x", "SMTP_PORT": "465"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert SMTPSettings.from_env().use_tls is False

    def test_dry_run_survives_missing_credentials(self):
        """A dry run must render without SMTP configured — that is its point."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert send_email("subj", "<p>x</p>", "x", dry_run=True) is False

    def test_a_real_send_still_demands_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EmailConfigError):
                send_email("subj", "<p>x</p>", "x")

    def test_message_is_multipart_with_text_first(self, projections, config):
        message_parts = render(projections, config, week=6, season=SEASON)
        settings = SMTPSettings(
            host="smtp.example.com", port=587, user="u", password="p",
            sender="me@example.com", recipients=["you@example.com"],
        )
        message = build_message(
            message_parts["subject"], message_parts["html"], message_parts["text"], settings
        )
        assert message.is_multipart()
        types = [part.get_content_type() for part in message.walk()]
        assert "text/plain" in types
        assert "text/html" in types
        assert message["To"] == "you@example.com"
        assert "Week 6" in message["Subject"]
