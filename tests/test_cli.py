"""End-to-end tests of the command line, with the API swapped for fixtures.

These cover the path that actually runs on a Thursday morning: resolve the
week, build every model, render, and hand off to the mailer.
"""

import datetime as dt
import os
from pathlib import Path
from unittest import mock

import pytest

from cfbmeta import cli
from cfbmeta.config import Config

from conftest import SEASON, FakeClient


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Point the CLI at fixtures and a scratch config."""
    monkeypatch.setattr(cli, "make_client", lambda config: FakeClient())

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "season: 2025\ntimezone: America/New_York\nsend_hour_local: 7\ncache_dir: ''\n"
    )
    return config_path


def run(args, patched):
    return cli.main(["--config", str(patched), *args])


class TestPreview:
    def test_writes_html(self, patched, tmp_path):
        out = tmp_path / "week6.html"
        assert run(["preview", "--week", "6", "--out", str(out)], patched) == 0
        html = out.read_text()
        assert "Week 6" in html
        assert len(html) > 5000

    def test_defaults_the_output_path(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert run(["preview", "--week", "6"], patched) == 0
        assert (tmp_path / "preview.html").exists()


class TestRun:
    def test_no_email_prints_the_text_report(self, patched, capsys):
        assert run(["run", "--week", "6", "--no-email"], patched) == 0
        out = capsys.readouterr().out
        assert "FULL SLATE" in out
        assert "Ohio State" in out

    def test_dry_run_does_not_send(self, patched, capsys, monkeypatch):
        sent = []
        monkeypatch.setenv("SMTP_USER", "me@example.com")
        monkeypatch.setenv("SMTP_PASSWORD", "app-password")

        import cfbmeta.email_send as email_send

        real_send = email_send.send_email

        def spy(subject, html, text, settings=None, dry_run=False):
            sent.append((subject, dry_run))
            return real_send(subject, html, text, settings, dry_run)

        monkeypatch.setattr(email_send, "send_email", spy)
        assert run(["run", "--week", "6", "--dry-run"], patched) == 0
        assert sent and sent[0][1] is True

    def test_time_gate_blocks_off_hour_runs(self, patched, monkeypatch, capsys):
        from zoneinfo import ZoneInfo

        wrong_hour = dt.datetime(2025, 10, 2, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        monkeypatch.setattr(cli, "local_now", lambda config: wrong_hour)
        assert run(["run", "--week", "6", "--check-time", "--no-email"], patched) == 0
        # Nothing rendered, because the gate exited first.
        assert "FULL SLATE" not in capsys.readouterr().out

    def test_time_gate_allows_the_send_hour(self, patched, monkeypatch, capsys):
        from zoneinfo import ZoneInfo

        right_hour = dt.datetime(2025, 10, 2, 7, 0, tzinfo=ZoneInfo("America/New_York"))
        monkeypatch.setattr(cli, "local_now", lambda config: right_hour)
        assert run(["run", "--week", "6", "--check-time", "--no-email"], patched) == 0
        assert "FULL SLATE" in capsys.readouterr().out

    def test_out_of_season_exits_cleanly(self, patched, monkeypatch, capsys):
        monkeypatch.setattr(cli, "resolve_week", lambda *a, **k: None)
        assert run(["run", "--no-email"], patched) == 0

    def test_missing_credentials_reports_cleanly(self, monkeypatch, tmp_path):
        # No fixture patch here: the real client should refuse without a key.
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        config_path = tmp_path / "config.yml"
        config_path.write_text("season: 2025\n")
        assert cli.main(["--config", str(config_path), "run", "--no-email"]) == 1


class TestBacktestCommand:
    def test_runs_and_reports(self, patched, capsys):
        assert run(["backtest", "--seasons", str(SEASON), "--basis", "same"], patched) == 0
        assert "Backtest over" in capsys.readouterr().out

    def test_refuses_to_write_config_from_contaminated_basis(self, patched, capsys):
        code = run(
            ["backtest", "--seasons", str(SEASON), "--basis", "same", "--fit", "--write"],
            patched,
        )
        assert code == 1
        assert "Refusing to write" in capsys.readouterr().out


class TestConfigLoading:
    def test_unknown_keys_are_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("not_a_real_setting: 1\n")
        with pytest.raises(ValueError, match="unknown config keys"):
            Config.load(path)

    def test_unknown_weight_keys_are_rejected(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("weights:\n  kenpom: 0.5\n")
        with pytest.raises(ValueError, match="unknown weight keys"):
            Config.load(path)

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yml"
        path.write_text("season: 2024\nsigma_margin: 16.0\n")
        monkeypatch.setenv("CFB_SEASON", "2025")
        monkeypatch.setenv("CFB_SIGMA", "14.5")
        config = Config.load(path)
        assert config.season == 2025
        assert config.sigma_margin == 14.5

    def test_missing_file_uses_defaults(self, tmp_path):
        config = Config.load(tmp_path / "nope.yml")
        assert config.timezone == "America/New_York"

    def test_season_inference(self):
        config = Config()
        config.season = 0
        assert config.resolved_season(dt.date(2025, 9, 1)) == 2025
        # January bowls still belong to the previous season.
        assert config.resolved_season(dt.date(2026, 1, 5)) == 2025

    def test_round_trips_through_save(self, tmp_path):
        path = tmp_path / "config.yml"
        original = Config()
        original.edge_shrink = 0.42
        original.weights.sp_plus = 0.5
        original.save(path)

        reloaded = Config.load(path)
        assert reloaded.edge_shrink == 0.42
        assert reloaded.weights.sp_plus == 0.5

    def test_shipped_config_is_valid(self):
        """The config.yml in the repo must actually load."""
        config = Config.load(Path(__file__).resolve().parent.parent / "config.yml")
        assert sum(config.weights.normalized().values()) == pytest.approx(1.0)
        assert 0.0 <= config.edge_shrink <= 1.0


class TestWeekZero:
    """Week 0 is a real slate (Ireland, the early kickoffs) and an easy one to
    break: it is falsy as an int and it sits before the first configured
    early-season entry.
    """

    def test_week_zero_is_not_swallowed_by_auto_detection(self, patched, tmp_path, monkeypatch):
        seen = {}
        real = cli.build_projections

        def spy(client, config, season, week):
            seen["week"] = week
            return real(client, config, season, week)

        monkeypatch.setattr(cli, "build_projections", spy)
        # If week 0 were treated as falsy, this would silently become week 6.
        run(["preview", "--week", "0", "--out", str(tmp_path / "w0.html")], patched)
        assert seen["week"] == 0

    def test_week_zero_downweights_the_in_season_sources(self):
        from cfbmeta.model import effective_weights

        config = Config.load(Path(__file__).resolve().parent.parent / "config.yml")
        week0 = effective_weights(config, 0)
        week1 = effective_weights(config, 1)
        late = effective_weights(config, 12)

        # Week 0 has even less current-season signal than week 1.
        assert week0["elo"] <= week1["elo"] < late["elo"]
        assert week0["srs"] <= week1["srs"] < late["srs"]
        assert week0["sp_plus"] >= week1["sp_plus"] > late["sp_plus"]

    def test_weeks_before_the_first_entry_clamp_rather_than_fall_through(self):
        config = Config()
        config.prior_weight_by_week = {1: 0.75, 2: 0.60}
        # Falling through to 0.0 would give Elo/SRS full weight in week 0.
        assert config.prior_weight(0) == pytest.approx(0.75)
        assert config.prior_weight(-1) == pytest.approx(0.75)
        assert config.prior_weight(2) == pytest.approx(0.60)
        assert config.prior_weight(9) == 0.0

    def test_empty_schedule_map_is_safe(self):
        config = Config()
        config.prior_weight_by_week = {}
        assert config.prior_weight(0) == 0.0


class TestDotenv:
    def test_reads_key_values(self, tmp_path, monkeypatch):
        from cfbmeta.config import load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text(
            "# a comment\n"
            "CFBD_API_KEY=abc123\n"
            "export SMTP_USER='me@example.com'\n"
            'EMAIL_TO="you@example.com"\n'
            "\n"
            "MALFORMED\n"
        )
        monkeypatch.delenv("CFBD_API_KEY", raising=False)
        monkeypatch.delenv("SMTP_USER", raising=False)
        monkeypatch.delenv("EMAIL_TO", raising=False)

        assert load_dotenv(env_file) == 3
        assert os.environ["CFBD_API_KEY"] == "abc123"
        assert os.environ["SMTP_USER"] == "me@example.com"
        assert os.environ["EMAIL_TO"] == "you@example.com"

    def test_real_environment_wins(self, tmp_path, monkeypatch):
        from cfbmeta.config import load_dotenv

        env_file = tmp_path / ".env"
        env_file.write_text("CFBD_API_KEY=from-file\n")
        monkeypatch.setenv("CFBD_API_KEY", "from-ci-secret")
        load_dotenv(env_file)
        # CI injects secrets properly; a stray local file must not override.
        assert os.environ["CFBD_API_KEY"] == "from-ci-secret"

    def test_missing_file_is_a_no_op(self, tmp_path):
        from cfbmeta.config import load_dotenv

        assert load_dotenv(tmp_path / "nope.env") == 0

    def test_dotenv_is_gitignored(self):
        ignored = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
        assert ".env" in ignored, "a committed .env would leak the API key"
