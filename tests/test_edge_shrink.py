"""The market-respect shrink is load-bearing for stake sizing, so it gets
its own tests: without it every disagreement reads as a maximum bet.
"""

import pytest

from cfbmeta.backtest import BacktestRow, fit_edge_shrink
from cfbmeta.probability import evaluate_spread_bet


def bet(projected, spread, **kwargs):
    kwargs.setdefault("min_edge_points", 1.5)
    return evaluate_spread_bet(projected, spread, 16.0, **kwargs)


class TestShrinkBehaviour:
    def test_raw_edge_is_reported_unshrunk(self):
        # The edge describes the disagreement honestly; only the stake shrinks.
        result = bet(10.0, -3.0, edge_shrink=0.35)
        assert result.edge_points == pytest.approx(7.0)

    def test_shrink_reduces_cover_probability(self):
        full = bet(10.0, -3.0, edge_shrink=1.0)
        shrunk = bet(10.0, -3.0, edge_shrink=0.35)
        assert full.cover_probability > shrunk.cover_probability
        assert shrunk.cover_probability > 0.5

    def test_shrink_reduces_stake(self):
        full = bet(10.0, -3.0, edge_shrink=1.0, max_units=100.0)
        shrunk = bet(10.0, -3.0, edge_shrink=0.35, max_units=100.0)
        assert shrunk.units < full.units

    def test_zero_shrink_means_the_market_is_always_right(self):
        result = bet(20.0, -3.0, edge_shrink=0.0)
        # With no signal credited, cover sits at a coin flip and EV goes
        # negative once the vig is paid, so nothing should be bet.
        assert result.cover_probability == pytest.approx(0.5, abs=0.02)
        assert result.side == "none"

    def test_shrink_keeps_the_side_it_picked(self):
        for shrink in (0.1, 0.35, 0.8, 1.0):
            assert bet(10.0, -3.0, edge_shrink=shrink).side in ("home", "none")
            assert bet(-10.0, 3.0, edge_shrink=shrink).side in ("away", "none")

    def test_out_of_range_shrink_is_clamped(self):
        assert bet(10.0, -3.0, edge_shrink=5.0).cover_probability == pytest.approx(
            bet(10.0, -3.0, edge_shrink=1.0).cover_probability
        )
        assert bet(10.0, -3.0, edge_shrink=-2.0).cover_probability == pytest.approx(
            bet(10.0, -3.0, edge_shrink=0.0).cover_probability
        )

    def test_realistic_edges_produce_realistic_probabilities(self):
        # A 3-point disagreement is a real but modest edge. Anything claiming
        # 60%+ from three points would be a red flag.
        result = bet(6.0, -3.0, edge_shrink=0.35)
        assert 0.50 < result.cover_probability < 0.56

    def test_stakes_stay_sane_across_the_slate(self):
        for edge in (2, 4, 6, 10, 20):
            result = bet(edge, 0.0, edge_shrink=0.35, max_units=100.0)
            assert result.units < 25.0, f"{edge}-point edge sized at {result.units}u"


class TestFitEdgeShrink:
    def _rows(self, signal: float, n: int = 600):
        """Synthetic games where our edge carries a known fraction of signal."""
        import random

        rng = random.Random(7)
        rows = []
        for _ in range(n):
            market = rng.uniform(-21, 21)
            disagreement = rng.uniform(-7, 7)
            actual = market + signal * disagreement + rng.gauss(0, 13)
            rows.append(
                BacktestRow(
                    season=2024, week=5, home_team="H", away_team="A",
                    actual_margin=actual,
                    component_margins={"sp_plus": market + disagreement},
                    hfa=0.0, adjustments=0.0, market_margin=market,
                )
            )
        return rows

    def test_recovers_a_known_signal_fraction(self):
        rows = self._rows(signal=0.4)
        fitted = fit_edge_shrink(rows, {"sp_plus": 1.0})
        assert fitted == pytest.approx(0.4, abs=0.12)

    def test_recovers_a_worthless_model(self):
        rows = self._rows(signal=0.0)
        fitted = fit_edge_shrink(rows, {"sp_plus": 1.0})
        assert fitted == pytest.approx(0.0, abs=0.12)

    def test_result_is_clamped_to_unit_interval(self):
        fitted = fit_edge_shrink(self._rows(signal=1.8), {"sp_plus": 1.0})
        assert 0.0 <= fitted <= 1.0

    def test_too_few_games_returns_none(self):
        assert fit_edge_shrink(self._rows(signal=0.4, n=50), {"sp_plus": 1.0}) is None

    def test_rows_without_lines_do_not_count_toward_the_minimum(self):
        # 600 rows, but only 150 have a line: not enough to fit anything.
        rows = self._rows(signal=0.4)
        for row in rows[:450]:
            row.market_margin = None
        assert fit_edge_shrink(rows, {"sp_plus": 1.0}) is None
