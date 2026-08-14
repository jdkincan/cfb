"""Backtesting and weight fitting.

This exists so the blend weights are measured rather than asserted. Ship
sensible priors, then let real results move them.

A caveat worth understanding before trusting any number this prints
------------------------------------------------------------------
CFBD serves *end-of-season* SP+, FPI and SRS ratings for a given year. Using
those to "predict" games from that same year is lookahead: the ratings already
know how the games turned out. That inflates accuracy and makes sigma look far
too small.

So the default mode (``basis="prior"``) rates every game using the **previous**
season's final ratings, which is information that genuinely existed before
kickoff. It understates in-season accuracy — real Thursday runs use current
ratings that have absorbed games up to that week — but it never flatters the
model, and it's the honest basis for estimating sigma.

``basis="same"`` uses same-season ratings. It's useful for comparing the
*relative* value of the sources to each other, since the lookahead contaminates
them all similarly, but the error figures it reports are not real. Anything
produced in that mode is labelled accordingly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import Config
from .model import IN_SEASON_SOURCES, SOURCE_LABELS, effective_weights
from .probability import DEFAULT_KEY_NUMBER_WEIGHTS, MarginDistribution, normal_cdf
from .ratings import ALL_SOURCES, RatingBook, build_rating_book, normalize_team
from .sources.cfbd import pick, pick_float
from .sources.market import build_market_map

log = logging.getLogger(__name__)


@dataclass
class BacktestRow:
    season: int
    week: int
    home_team: str
    away_team: str
    actual_margin: float
    component_margins: Dict[str, float]
    hfa: float
    adjustments: float
    market_margin: Optional[float] = None

    def projected(self, weights: Dict[str, float]) -> float:
        num = den = 0.0
        for source, margin in self.component_margins.items():
            w = weights.get(source, 0.0)
            if w > 0:
                num += margin * w
                den += w
        if den <= 0:
            return self.hfa + self.adjustments
        return num / den + self.hfa + self.adjustments


@dataclass
class BacktestResult:
    rows: List[BacktestRow] = field(default_factory=list)
    basis: str = "prior"
    weights: Dict[str, float] = field(default_factory=dict)
    mae: float = 0.0
    rmse: float = 0.0
    sigma: float = 0.0
    bias: float = 0.0
    market_mae: Optional[float] = None
    ats_wins: int = 0
    ats_losses: int = 0
    ats_pushes: int = 0
    calibration: List[Tuple[str, int, float, float]] = field(default_factory=list)
    per_source: Dict[str, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def ats_pct(self) -> Optional[float]:
        graded = self.ats_wins + self.ats_losses
        return self.ats_wins / graded if graded else None

    @property
    def trustworthy(self) -> bool:
        return self.basis == "prior"

    def summary(self) -> str:
        lines = [
            f"Backtest over {self.n} games (basis: {self.basis}-season ratings)",
        ]
        if not self.trustworthy:
            lines.append(
                "  !! same-season ratings include lookahead; error figures below "
                "are optimistic and must not be used to set sigma."
            )
        lines += [
            f"  MAE          {self.mae:6.2f} points",
            f"  RMSE         {self.rmse:6.2f} points",
            f"  sigma        {self.sigma:6.2f} points",
            f"  mean bias    {self.bias:+6.2f} points (positive = favors home)",
        ]
        if self.market_mae is not None:
            delta = self.mae - self.market_mae
            verdict = "better than" if delta < 0 else "worse than"
            lines.append(
                f"  market MAE   {self.market_mae:6.2f} points "
                f"({abs(delta):.2f} {verdict} the closing line)"
            )
        if self.ats_pct is not None:
            lines.append(
                f"  ATS          {self.ats_wins}-{self.ats_losses}-{self.ats_pushes} "
                f"({self.ats_pct:.1%}, break-even 52.4%)"
            )
        if self.weights:
            pretty = ", ".join(
                f"{SOURCE_LABELS.get(k, k)} {v:.3f}"
                for k, v in sorted(self.weights.items(), key=lambda kv: -kv[1])
                if v > 0.0005
            )
            lines.append(f"  weights      {pretty}")
        if self.per_source:
            lines.append("  standalone MAE by source:")
            for source, mae in sorted(self.per_source.items(), key=lambda kv: kv[1]):
                lines.append(f"    {SOURCE_LABELS.get(source, source):8s} {mae:6.2f}")
        if self.calibration:
            lines.append("  calibration (predicted vs actual home win rate):")
            for label, count, predicted, actual in self.calibration:
                lines.append(
                    f"    {label:>12s}  n={count:4d}  predicted {predicted:.1%}  "
                    f"actual {actual:.1%}"
                )
        return "\n".join(lines)


def collect_rows(
    client,
    config: Config,
    seasons: Sequence[int],
    basis: str = "prior",
    hfa_model=None,
    coach_model=None,
    situational=None,
) -> List[BacktestRow]:
    """Build one row per completed game with each source's neutral margin."""
    rows: List[BacktestRow] = []

    for season in seasons:
        rating_season = season - 1 if basis == "prior" else season
        try:
            book = build_rating_book(client, rating_season)
        except Exception as exc:  # noqa: BLE001
            log.warning("no ratings for %d: %s", rating_season, exc)
            continue
        if not len(book):
            continue

        try:
            games = client.games(season)
        except Exception as exc:  # noqa: BLE001
            log.warning("no games for %d: %s", season, exc)
            continue

        markets: Dict = {}
        try:
            markets = build_market_map(client.lines(season))
        except Exception as exc:  # noqa: BLE001
            log.debug("no lines for %d: %s", season, exc)

        for game in games:
            home = pick(game, "homeTeam", "home_team")
            away = pick(game, "awayTeam", "away_team")
            hp = pick_float(game, "homePoints", "home_points")
            ap = pick_float(game, "awayPoints", "away_points")
            if not home or not away or hp is None or ap is None:
                continue

            margins = {}
            for source in ALL_SOURCES:
                h = book.rating(home, source)
                a = book.rating(away, source)
                if h is not None and a is not None:
                    margins[source] = h - a
            if not margins:
                continue

            neutral = bool(pick(game, "neutralSite", "neutral_site", default=False))
            hfa = 0.0
            if not neutral:
                hfa = (
                    hfa_model.for_game(home, away, neutral)["total"]
                    if hfa_model
                    else config.league_hfa
                )

            adjustments = 0.0
            if coach_model is not None:
                adjustments += float(coach_model.for_game(home, away).get("total", 0.0))

            market = markets.get(pick(game, "id")) or markets.get(
                (normalize_team(home), normalize_team(away))
            )
            market_margin = None
            if market and market.get("spread") is not None:
                market_margin = -float(market["spread"])

            rows.append(
                BacktestRow(
                    season=season,
                    week=int(pick_float(game, "week", default=0) or 0),
                    home_team=home,
                    away_team=away,
                    actual_margin=hp - ap,
                    component_margins=margins,
                    hfa=hfa,
                    adjustments=adjustments,
                    market_margin=market_margin,
                )
            )

    return rows


def fit_weights(rows: List[BacktestRow], sources: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Least-squares blend weights, constrained to be non-negative and sum to 1.

    Solves for the weights that best explain ``actual - hfa - adjustments`` from
    the component margins, then clips negatives to zero and renormalizes. A
    negative weight would mean "bet against this rating system", which is
    almost always overfitting rather than signal, so it gets clipped instead.
    """
    import numpy as np

    sources = list(sources or ALL_SOURCES)
    usable = [r for r in rows if all(s in r.component_margins for s in sources)]
    if len(usable) < 50:
        log.warning("only %d complete rows; keeping prior weights", len(usable))
        return {}

    X = np.array([[r.component_margins[s] for s in sources] for r in usable], dtype=float)
    y = np.array([r.actual_margin - r.hfa - r.adjustments for r in usable], dtype=float)

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.clip(coef, 0.0, None)
    total = coef.sum()
    if total <= 0:
        log.warning("degenerate weight fit; keeping prior weights")
        return {}
    coef = coef / total
    return {s: round(float(c), 4) for s, c in zip(sources, coef)}


def fit_edge_shrink(rows: List[BacktestRow], weights: Dict[str, float]) -> Optional[float]:
    """Estimate how much of our disagreement with the market is real signal.

    Regress the actual margin on the market line and on our disagreement:

        actual = a * market_margin + b * (projected - market_margin)

    ``b`` is the answer. If our extra information were worthless, ``b`` would be
    0 and every bet should be passed; if we were strictly sharper than the
    closing line, ``b`` would approach 1. For a model assembled from public
    ratings, values in the 0.2-0.4 range are typical and anything near 1 should
    be read as a bug or as lookahead contamination rather than as an edge.
    """
    import numpy as np

    usable = [r for r in rows if r.market_margin is not None]
    if len(usable) < 200:
        log.warning("only %d games with lines; keeping the configured edge shrink", len(usable))
        return None

    X = np.array(
        [[r.market_margin, r.projected(weights) - r.market_margin] for r in usable], dtype=float
    )
    y = np.array([r.actual_margin for r in usable], dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    shrink = float(coef[1])
    log.info(
        "edge-shrink fit: market coefficient %.3f, disagreement coefficient %.3f",
        float(coef[0]), shrink,
    )
    return round(max(0.0, min(1.0, shrink)), 3)


def evaluate(
    rows: List[BacktestRow],
    weights: Dict[str, float],
    basis: str = "prior",
    sigma_hint: Optional[float] = None,
) -> BacktestResult:
    """Score a weight set: error, bias, ATS record and calibration."""
    result = BacktestResult(rows=rows, basis=basis, weights=dict(weights))
    if not rows:
        return result

    errors, market_errors = [], []
    for row in rows:
        errors.append(row.actual_margin - row.projected(weights))
        if row.market_margin is not None:
            market_errors.append(row.actual_margin - row.market_margin)

    n = len(errors)
    result.mae = round(sum(abs(e) for e in errors) / n, 3)
    result.rmse = round(math.sqrt(sum(e * e for e in errors) / n), 3)
    result.bias = round(-sum(errors) / n, 3)
    mean_err = sum(errors) / n
    result.sigma = round(
        math.sqrt(sum((e - mean_err) ** 2 for e in errors) / max(1, n - 1)), 3
    )
    if market_errors:
        result.market_mae = round(sum(abs(e) for e in market_errors) / len(market_errors), 3)

    # Standalone accuracy per source, for the report.
    for source in ALL_SOURCES:
        subset = [r for r in rows if source in r.component_margins]
        if len(subset) < 50:
            continue
        errs = [
            r.actual_margin - (r.component_margins[source] + r.hfa + r.adjustments)
            for r in subset
        ]
        result.per_source[source] = round(sum(abs(e) for e in errs) / len(errs), 3)

    sigma = sigma_hint or result.sigma or 16.0
    _grade_ats(result, rows, weights, sigma)
    _calibrate(result, rows, weights, sigma)
    return result


def _grade_ats(
    result: BacktestResult, rows: List[BacktestRow], weights: Dict[str, float], sigma: float
) -> None:
    """How the model's side would have done against the closing number."""
    for row in rows:
        if row.market_margin is None:
            continue
        projected = row.projected(weights)
        edge = projected - row.market_margin
        if abs(edge) < 1.5:  # only grade what we'd actually have bet
            continue
        if row.actual_margin == row.market_margin:
            result.ats_pushes += 1
        elif (edge > 0) == (row.actual_margin > row.market_margin):
            result.ats_wins += 1
        else:
            result.ats_losses += 1


def _calibrate(
    result: BacktestResult, rows: List[BacktestRow], weights: Dict[str, float], sigma: float
) -> None:
    """Bucket predicted home win probability against what actually happened."""
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    tallies = {b: [0, 0.0, 0] for b in buckets}  # count, predicted sum, wins

    for row in rows:
        p = MarginDistribution(row.projected(weights), sigma).p_home_win()
        for lo, hi in buckets:
            if lo <= p < hi or (hi == 1.0 and p == 1.0):
                tally = tallies[(lo, hi)]
                tally[0] += 1
                tally[1] += p
                tally[2] += 1 if row.actual_margin > 0 else 0
                break

    for (lo, hi), (count, psum, wins) in tallies.items():
        if count:
            result.calibration.append(
                (f"{lo:.0%}-{hi:.0%}", count, psum / count, wins / count)
            )


def fit_key_numbers(rows: List[BacktestRow], sigma: float = 16.0) -> Dict[int, float]:
    """Derive key-number multipliers from the empirical margin distribution.

    For each margin, compare how often it actually happened against how often a
    plain normal says it should have. The ratio is the multiplier.
    """
    if len(rows) < 500:
        log.warning("only %d games; keeping default key numbers", len(rows))
        return dict(DEFAULT_KEY_NUMBER_WEIGHTS)

    observed: Dict[int, int] = {}
    for row in rows:
        m = int(round(abs(row.actual_margin)))
        if m > 0:
            observed[m] = observed.get(m, 0) + 1

    total = sum(observed.values())
    mean_abs = sum(abs(r.actual_margin) for r in rows) / len(rows)
    weights: Dict[int, float] = {}
    for margin, count in observed.items():
        if margin > 35 or count < 10:
            continue
        actual_rate = count / total
        # Expected rate under a half-normal with the observed scale.
        expected_rate = 2.0 * (
            normal_cdf((margin + 0.5) / (mean_abs * 1.2533))
            - normal_cdf((margin - 0.5) / (mean_abs * 1.2533))
        )
        if expected_rate <= 0:
            continue
        weights[margin] = round(min(3.0, max(0.4, actual_rate / expected_rate)), 3)
    return weights or dict(DEFAULT_KEY_NUMBER_WEIGHTS)


def run_backtest(
    client,
    config: Config,
    seasons: Sequence[int],
    basis: str = "prior",
    fit: bool = False,
    hfa_model=None,
    coach_model=None,
) -> BacktestResult:
    """Collect, optionally refit, and evaluate."""
    rows = collect_rows(
        client, config, seasons, basis=basis, hfa_model=hfa_model, coach_model=coach_model
    )
    if not rows:
        return BacktestResult(basis=basis)

    weights = config.weights.normalized()
    weights.pop("market", None)
    if fit:
        fitted = fit_weights(rows)
        if fitted:
            weights = fitted
            log.info("fitted weights: %s", fitted)

    return evaluate(rows, weights, basis=basis)
