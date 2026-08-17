"""Cross-run staleness detection.

The source audit catches a missing source and a flat source from a single run.
It cannot catch the third failure — real, well-dispersed, *outdated* values —
because a stale feed looks identical to a fresh one in isolation. That needs
memory across runs.
"""

import datetime as dt

import pytest

from cfbmeta.freshness import fingerprint, load_state, record, save_state
from cfbmeta.ratings import SourceStatus

NOW = dt.datetime(2026, 10, 1, 12, tzinfo=dt.timezone.utc)


class TestFingerprint:
    def test_same_values_hash_the_same(self):
        assert fingerprint({"a": 1.0, "b": 2.0}) == fingerprint({"b": 2.0, "a": 1.0})

    def test_changed_values_hash_differently(self):
        assert fingerprint({"a": 1.0}) != fingerprint({"a": 1.1})

    def test_added_team_changes_the_hash(self):
        assert fingerprint({"a": 1.0}) != fingerprint({"a": 1.0, "b": 2.0})

    def test_tiny_changes_are_detected(self):
        # Ratings are published to a decimal; a real revision must register.
        assert fingerprint({"a": 28.70}) != fingerprint({"a": 28.71})


class TestRecord:
    def test_first_sighting_is_zero_days(self):
        state = {}
        assert record("2026:sp_plus", "abc", state, NOW) == 0.0
        assert state["2026:sp_plus"]["fingerprint"] == "abc"

    def test_unchanged_values_accumulate_age(self):
        state = {}
        record("2026:sp_plus", "abc", state, NOW)
        days = record("2026:sp_plus", "abc", state, NOW + dt.timedelta(days=9))
        assert days == pytest.approx(9.0, abs=0.01)

    def test_a_change_resets_the_clock(self):
        state = {}
        record("2026:sp_plus", "abc", state, NOW)
        record("2026:sp_plus", "abc", state, NOW + dt.timedelta(days=9))
        assert record("2026:sp_plus", "xyz", state, NOW + dt.timedelta(days=10)) == 0.0

    def test_sources_are_tracked_independently(self):
        state = {}
        record("2026:sp_plus", "abc", state, NOW)
        record("2026:fpi", "def", state, NOW)
        later = NOW + dt.timedelta(days=8)
        record("2026:sp_plus", "abc", state, later)
        assert record("2026:fpi", "NEW", state, later) == 0.0

    def test_seasons_are_tracked_independently(self):
        state = {}
        record("2025:sp_plus", "abc", state, NOW)
        assert record("2026:sp_plus", "abc", state, NOW) == 0.0

    def test_corrupt_timestamp_does_not_crash(self):
        state = {"2026:sp_plus": {"fingerprint": "abc", "first_seen": "not-a-date"}}
        assert record("2026:sp_plus", "abc", state, NOW) == 0.0


class TestStaleFlag:
    def test_fresh_source_is_not_stale(self):
        status = SourceStatus("sp_plus", 2026, teams=136, usable=True, unchanged_days=2.0)
        assert status.stale is False
        assert "STALE" not in status.describe()

    def test_week_old_values_are_flagged(self):
        status = SourceStatus("sp_plus", 2026, teams=136, usable=True, unchanged_days=14.0)
        assert status.stale is True
        assert "STALE" in status.describe()

    def test_an_unusable_source_is_not_also_called_stale(self):
        status = SourceStatus("srs", 2026, usable=False, reason="no data", unchanged_days=99)
        assert status.stale is False

    def test_age_appears_in_the_description(self):
        status = SourceStatus("fpi", 2026, teams=138, usable=True,
                              dispersion=11.4, unchanged_days=3.0)
        assert "unchanged 3d" in status.describe()

    def test_same_day_age_is_not_reported(self):
        status = SourceStatus("fpi", 2026, teams=138, usable=True, unchanged_days=0.2)
        assert "unchanged" not in status.describe()


class TestStatePersistence:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "state.json"
        state = {}
        record("2026:sp_plus", "abc", state, NOW)
        save_state(state, path)
        assert load_state(path)["2026:sp_plus"]["fingerprint"] == "abc"

    def test_missing_file_is_empty(self, tmp_path):
        assert load_state(tmp_path / "nope.json") == {}

    def test_corrupt_file_is_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json")
        assert load_state(path) == {}
