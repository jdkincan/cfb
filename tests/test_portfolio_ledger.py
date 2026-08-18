"""Portfolio sizing, the bet ledger, availability overrides, and best lines."""

import pytest

from cfbmeta.availability import MAX_PER_TEAM, Adjustment, AvailabilityModel, load
from cfbmeta.ledger import Bet, append, load as load_ledger, settle, summarize
from cfbmeta.portfolio import Position, apply, effective_bet_count, resize
from cfbmeta.sources.market import best_spread, consensus_line


def positions(n, conference="SEC", bucket="2026-09-05", units=1.5):
    return [Position(key=str(i), label=f"bet{i}", units=units,
                     conference=conference, kickoff_bucket=bucket) for i in range(n)]


class TestCorrelation:
    def test_identical_group_collapses_toward_one_bet(self):
        assert effective_bet_count(positions(8)) < 3.0

    def test_spread_across_groups_stays_independent(self):
        mixed = [Position(key=str(i), label="x", units=1.0, conference=f"C{i}",
                          kickoff_bucket=f"d{i}") for i in range(8)]
        assert effective_bet_count(mixed) == pytest.approx(8.0)

    def test_single_bet_is_one(self):
        assert effective_bet_count(positions(1)) == pytest.approx(1.0)

    def test_empty_card_is_zero(self):
        assert effective_bet_count([]) == 0.0

    def test_correlation_scales_the_card_down(self):
        result = apply(positions(8), max_weekly_units=100.0)
        assert result.correlation_factor < 1.0
        assert result.scaled_units < result.raw_units

    def test_uncorrelated_card_is_barely_touched(self):
        mixed = [Position(key=str(i), label="x", units=1.0, conference=f"C{i}",
                          kickoff_bucket=f"d{i}") for i in range(6)]
        assert apply(mixed, max_weekly_units=100.0).correlation_factor == pytest.approx(1.0)


class TestExposure:
    def test_total_exposure_is_capped(self):
        result = apply(positions(6, units=3.0), max_weekly_units=5.0)
        assert result.scaled_units <= 5.0 + 1e-6

    def test_per_play_cap_still_binds(self):
        result = apply(positions(2, conference="", bucket="", units=9.0),
                       max_weekly_units=100.0, max_units_per_play=3.0)
        assert all(p.units <= 3.0 for p in result.positions)

    def test_small_card_is_not_scaled_up(self):
        """Being under the exposure budget is not a reason to bet more."""
        result = apply(positions(1, units=1.0), max_weekly_units=50.0)
        assert result.scaled_units == pytest.approx(1.0)


class TestResize:
    def test_keeps_only_the_chosen_plays(self):
        result = resize(positions(6), keep=["0", "2"], max_weekly_units=100.0)
        assert {p.key for p in result.positions} == {"0", "2"}

    def test_a_smaller_card_is_less_diversified_so_scales_up_per_bet(self):
        full = apply(positions(8), max_weekly_units=100.0)
        few = resize(positions(8), keep=["0", "1"], max_weekly_units=100.0)
        # Fewer correlated bets means a milder correlation haircut per bet.
        assert few.correlation_factor > full.correlation_factor

    def test_stakes_are_not_inflated_to_refill_the_budget(self):
        few = resize(positions(8), keep=["0"], max_weekly_units=100.0)
        assert few.positions[0].units <= 1.5

    def test_unknown_ids_are_ignored(self):
        result = resize(positions(3), keep=["0", "nope"], max_weekly_units=100.0)
        assert len(result.positions) == 1


class TestAvailability:
    def test_adjustment_applies_to_the_named_team(self):
        model = AvailabilityModel(adjustments={"arkansas": Adjustment("Arkansas", -6.5, "QB")})
        assert model.for_game("Arkansas", "LSU")["total"] == pytest.approx(-6.5)
        assert model.for_game("LSU", "Arkansas")["total"] == pytest.approx(6.5)

    def test_reason_is_carried_through(self):
        model = AvailabilityModel(adjustments={"arkansas": Adjustment("Arkansas", -6.5, "QB out")})
        assert model.for_game("Arkansas", "LSU")["home_reason"] == "QB out"

    def test_absurd_entries_are_capped(self):
        model = AvailabilityModel(adjustments={"x": Adjustment("X", -99.0)})
        assert model.points_for("X") == pytest.approx(-MAX_PER_TEAM)

    def test_unknown_team_is_neutral(self):
        assert AvailabilityModel().points_for("Anyone") == 0.0

    def test_entries_expire_when_the_week_moves_on(self, tmp_path):
        path = tmp_path / "availability.yml"
        path.write_text(
            "2026:\n  week: 3\n  adjustments:\n"
            "    - team: Arkansas\n      points: -6.5\n      reason: QB out\n"
        )
        assert load(2026, 3, path).points_for("Arkansas") == pytest.approx(-6.5)
        # Week 4 must not inherit week 3's injuries.
        assert load(2026, 4, path).points_for("Arkansas") == 0.0

    def test_missing_file_is_safe(self, tmp_path):
        assert load(2026, 1, tmp_path / "nope.yml").adjustments == {}

    def test_malformed_file_is_safe(self, tmp_path):
        path = tmp_path / "availability.yml"
        path.write_text("not: [valid")
        assert load(2026, 1, path).adjustments == {}


class TestBestLine:
    QUOTES = [
        {"provider": "DraftKings", "spread": -7.0, "spreadOpen": -6.5, "overUnder": 52.5},
        {"provider": "ESPN Bet", "spread": -7.5, "spreadOpen": -6.5, "overUnder": 52.5},
        {"provider": "Bovada", "spread": -6.5, "spreadOpen": -7.0, "overUnder": 53.0},
    ]

    def test_home_bettor_wants_the_largest_spread(self):
        best = best_spread(self.QUOTES, "home")
        assert best["spread"] == pytest.approx(-6.5)
        assert best["provider"] == "Bovada"

    def test_away_bettor_wants_the_smallest(self):
        best = best_spread(self.QUOTES, "away")
        assert best["spread"] == pytest.approx(-7.5)
        assert best["provider"] == "ESPN Bet"

    def test_ties_break_on_book_preference(self):
        quotes = [{"provider": "Bovada", "spread": -7.0},
                  {"provider": "DraftKings", "spread": -7.0}]
        assert best_spread(quotes, "home")["provider"] == "DraftKings"

    def test_no_quotes_is_none(self):
        assert best_spread([], "home") is None

    def test_consensus_reports_movement_from_open(self):
        line = consensus_line({"lines": self.QUOTES})
        assert line["spread"] == pytest.approx(-7.0)
        assert line["spread_open"] == pytest.approx(-6.5)
        # Number moved from -6.5 to -7.0: further toward the home favourite.
        assert line["movement"] == pytest.approx(0.5)

    def test_consensus_carries_both_best_sides(self):
        line = consensus_line({"lines": self.QUOTES})
        assert line["best_home"]["spread"] == pytest.approx(-6.5)
        assert line["best_away"]["spread"] == pytest.approx(-7.5)


class TestLedger:
    def _bet(self, **kw):
        base = dict(season=2026, week=1, game_id="1", matchup="A at B", side="home",
                    team="B", line_taken=-3.5, units=1.5, book="DraftKings")
        base.update(kw)
        return Bet(**base)

    def test_append_and_read_back(self, tmp_path):
        path = tmp_path / "bets.csv"
        assert append([self._bet()], path) == 1
        assert len(load_ledger(path)) == 1

    def test_duplicates_are_not_double_recorded(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet()], path)
        assert append([self._bet()], path) == 0

    def test_a_winning_home_bet_settles_correctly(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet()], path)
        settle({"1": 7.0}, path=path)
        row = load_ledger(path)[0]
        assert row["result"] == "win"
        assert float(row["profit_units"]) == pytest.approx(1.5 * 100 / 110, abs=0.01)

    def test_a_losing_bet_costs_the_stake(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet()], path)
        settle({"1": 1.0}, path=path)
        row = load_ledger(path)[0]
        assert row["result"] == "loss"
        assert float(row["profit_units"]) == pytest.approx(-1.5)

    def test_an_exact_number_is_a_push(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet(line_taken=-7.0)], path)
        settle({"1": 7.0}, path=path)
        row = load_ledger(path)[0]
        assert row["result"] == "push"
        assert float(row["profit_units"]) == 0.0

    def test_away_side_grades_the_other_way(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet(side="away", team="A", line_taken=3.5)], path)
        settle({"1": 7.0}, path=path)
        assert load_ledger(path)[0]["result"] == "loss"

    def test_clv_is_positive_when_the_number_beat_the_close(self, tmp_path):
        path = tmp_path / "bets.csv"
        # Took home -3.5; it closed -5.0, so the number taken was better.
        append([self._bet()], path)
        settle({"1": 7.0}, closing={"1": -5.0}, path=path)
        assert float(load_ledger(path)[0]["clv_points"]) == pytest.approx(1.5)

    def test_clv_is_negative_when_the_number_got_worse(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet()], path)
        settle({"1": 7.0}, closing={"1": -2.0}, path=path)
        assert float(load_ledger(path)[0]["clv_points"]) == pytest.approx(-1.5)

    def test_summary_reports_roi_and_clv(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet(), self._bet(game_id="2", side="away", team="A", line_taken=3.5)], path)
        settle({"1": 7.0, "2": 7.0}, closing={"1": -5.0, "2": 5.0}, path=path)
        summary = summarize(path)
        assert summary.settled == 2
        assert summary.wins == 1 and summary.losses == 1
        assert summary.clv_sample == 2
        assert "CLV" in summary.describe()

    def test_settling_twice_does_not_double_count(self, tmp_path):
        path = tmp_path / "bets.csv"
        append([self._bet()], path)
        settle({"1": 7.0}, path=path)
        assert settle({"1": 7.0}, path=path) == 0

    def test_empty_ledger_summarizes_cleanly(self, tmp_path):
        assert summarize(tmp_path / "none.csv").bets == 0
