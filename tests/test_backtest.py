import random

import pytest

from cfbmeta.backtest import (
    BacktestRow,
    collect_rows,
    evaluate,
    fit_key_numbers,
    fit_weights,
    run_backtest,
)
from cfbmeta.config import Config

from conftest import SEASON


def make_rows(n=400, noise=13.0, seed=3):
    """Games where the truth is 0.6*SP+ + 0.4*FPI plus home field."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        sp = rng.uniform(-28, 28)
        fpi = sp + rng.gauss(0, 4)
        truth = 0.6 * sp + 0.4 * fpi + 2.5
        rows.append(
            BacktestRow(
                season=2024, week=rng.randint(1, 13), home_team="H", away_team="A",
                actual_margin=truth + rng.gauss(0, noise),
                component_margins={"sp_plus": sp, "fpi": fpi},
                hfa=2.5, adjustments=0.0,
                market_margin=truth + rng.gauss(0, 2),
            )
        )
    return rows


class TestRowProjection:
    def test_projection_applies_weights_and_adjustments(self):
        row = BacktestRow(
            season=2024, week=1, home_team="H", away_team="A", actual_margin=0.0,
            component_margins={"sp_plus": 10.0, "fpi": 20.0},
            hfa=3.0, adjustments=1.0,
        )
        assert row.projected({"sp_plus": 0.5, "fpi": 0.5}) == pytest.approx(19.0)

    def test_weights_renormalize_over_present_sources(self):
        row = BacktestRow(
            season=2024, week=1, home_team="H", away_team="A", actual_margin=0.0,
            component_margins={"sp_plus": 10.0},
            hfa=0.0, adjustments=0.0,
        )
        # FPI is weighted but absent, so SP+ should carry the whole projection.
        assert row.projected({"sp_plus": 0.4, "fpi": 0.6}) == pytest.approx(10.0)

    def test_no_sources_falls_back_to_adjustments(self):
        row = BacktestRow(
            season=2024, week=1, home_team="H", away_team="A", actual_margin=0.0,
            component_margins={}, hfa=2.5, adjustments=1.0,
        )
        assert row.projected({"sp_plus": 1.0}) == pytest.approx(3.5)


class TestFitWeights:
    def test_recovers_the_generating_weights(self):
        fitted = fit_weights(make_rows(1200), sources=["sp_plus", "fpi"])
        assert sum(fitted.values()) == pytest.approx(1.0)
        # Noisy and collinear, so this is a loose check on the ordering.
        assert fitted["sp_plus"] > 0.2

    def test_weights_are_non_negative(self):
        fitted = fit_weights(make_rows(800), sources=["sp_plus", "fpi"])
        assert all(v >= 0 for v in fitted.values())

    def test_too_few_rows_keeps_the_priors(self):
        assert fit_weights(make_rows(20), sources=["sp_plus", "fpi"]) == {}

    def test_rows_missing_a_source_are_skipped(self):
        rows = make_rows(60)
        for row in rows:
            row.component_margins.pop("fpi", None)
        assert fit_weights(rows, sources=["sp_plus", "fpi"]) == {}


class TestEvaluate:
    def test_reports_error_metrics(self):
        result = evaluate(make_rows(500), {"sp_plus": 0.6, "fpi": 0.4})
        assert result.n == 500
        assert result.mae > 0
        assert result.rmse >= result.mae
        assert result.sigma > 0

    def test_market_comparison_is_included(self):
        result = evaluate(make_rows(500), {"sp_plus": 0.6, "fpi": 0.4})
        assert result.market_mae is not None
        assert result.market_mae > 0

    def test_a_worse_model_loses_to_the_market(self):
        """The market comparison has to be able to say we lost."""
        rows = make_rows(1500)
        # Ignore FPI entirely and add a systematic bias: strictly worse than
        # the near-perfect synthetic market line.
        for row in rows:
            row.adjustments = 6.0
        result = evaluate(rows, {"sp_plus": 1.0})
        assert result.market_mae < result.mae
        assert "worse than" in result.summary()

    def test_a_better_model_beats_the_market(self):
        result = evaluate(make_rows(1500), {"sp_plus": 0.6, "fpi": 0.4})
        assert "better than" in result.summary()

    def test_ats_record_is_graded(self):
        result = evaluate(make_rows(500), {"sp_plus": 0.6, "fpi": 0.4})
        assert result.ats_wins + result.ats_losses + result.ats_pushes > 0

    def test_calibration_buckets_are_produced(self):
        result = evaluate(make_rows(500), {"sp_plus": 0.6, "fpi": 0.4})
        assert result.calibration
        for _, count, predicted, actual in result.calibration:
            assert count > 0
            assert 0.0 <= predicted <= 1.0
            assert 0.0 <= actual <= 1.0

    def test_bias_is_near_zero_for_an_unbiased_model(self):
        result = evaluate(make_rows(2000), {"sp_plus": 0.6, "fpi": 0.4})
        assert abs(result.bias) < 1.5

    def test_summary_is_printable(self):
        summary = evaluate(make_rows(300), {"sp_plus": 0.6, "fpi": 0.4}).summary()
        assert "MAE" in summary and "sigma" in summary

    def test_empty_input_is_safe(self):
        result = evaluate([], {"sp_plus": 1.0})
        assert result.n == 0
        assert "0 games" in result.summary()


class TestLookaheadGuard:
    def test_same_season_basis_is_flagged_as_untrustworthy(self):
        result = evaluate(make_rows(300), {"sp_plus": 1.0}, basis="same")
        assert result.trustworthy is False
        assert "lookahead" in result.summary()

    def test_prior_season_basis_is_trusted(self):
        result = evaluate(make_rows(300), {"sp_plus": 1.0}, basis="prior")
        assert result.trustworthy is True
        assert "lookahead" not in result.summary()


class TestKeyNumbers:
    def test_small_samples_keep_the_defaults(self):
        from cfbmeta.probability import DEFAULT_KEY_NUMBER_WEIGHTS

        assert fit_key_numbers(make_rows(100)) == DEFAULT_KEY_NUMBER_WEIGHTS

    def test_key_numbers_emerge_from_a_spiked_distribution(self):
        """Feed in games that land on 3 and 7 far too often; expect a lift."""
        rng = random.Random(11)
        rows = []
        for i in range(3000):
            margin = rng.gauss(0, 15)
            if i % 5 == 0:
                margin = rng.choice([3, -3, 7, -7])
            rows.append(
                BacktestRow(
                    season=2024, week=1, home_team="H", away_team="A",
                    actual_margin=margin, component_margins={"sp_plus": 0.0},
                    hfa=0.0, adjustments=0.0,
                )
            )
        weights = fit_key_numbers(rows)
        assert weights[3] > 1.2
        assert weights[7] > 1.2


class TestCollectRows:
    def test_collects_completed_games_only(self, client, config):
        rows = collect_rows(client, config, [SEASON], basis="same")
        assert rows
        # Week 6 in the fixture has no scores, so it must not appear.
        assert all(row.week != 6 for row in rows)

    def test_prior_basis_uses_the_previous_season_ratings(self, client, config):
        client.calls.clear()
        collect_rows(client, config, [SEASON], basis="prior")
        assert "sp" in client.calls

    def test_each_row_carries_component_margins(self, client, config):
        rows = collect_rows(client, config, [SEASON], basis="same")
        assert all(row.component_margins for row in rows)

    def test_run_backtest_end_to_end(self, client, config):
        result = run_backtest(client, config, [SEASON], basis="same")
        assert result.n > 0
        assert result.mae > 0
