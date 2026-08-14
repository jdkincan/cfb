"""Turn a projected margin into probabilities, edges and stake sizes.

Sign conventions, fixed everywhere in this codebase:

* ``margin`` always means ``home_points - away_points``. Positive = home won.
* ``spread`` follows the sportsbook/CFBD convention: quoted from the home
  team's side, negative when the home team is favored. ``-7`` = home by 7.
* ``market_margin`` is ``-spread`` — the home margin the market expects. It is
  the directly comparable number to our projected margin.
* ``edge`` is ``projected_margin - market_margin``. Positive means we like the
  home side relative to the market; negative means we like the away side.

College football margins are not smoothly normal: they pile up on 3 and 7 and
to a lesser degree 10, 14, 17 and 21. A plain normal therefore misprices any
bet sitting on a key number. :class:`MarginDistribution` discretizes the normal
and reweights those integers. Defaults below come from published FBS margin
frequencies; ``cfbmeta backtest --fit-key-numbers`` re-derives them from actual
results and writes them to ``calibration/key_numbers.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

DEFAULT_KEY_NUMBER_WEIGHTS: Dict[int, float] = {
    1: 0.95,
    2: 0.95,
    3: 1.95,
    4: 1.05,
    5: 0.90,
    6: 1.00,
    7: 1.55,
    8: 0.90,
    9: 0.95,
    10: 1.25,
    11: 0.90,
    13: 0.95,
    14: 1.20,
    16: 0.95,
    17: 1.15,
    18: 0.95,
    20: 1.05,
    21: 1.15,
    24: 1.05,
    28: 1.05,
}

# Lives outside data/ because data/ is gitignored for caches, and fitted key
# numbers need to be committed to reach the scheduled run.
KEY_NUMBERS_PATH = Path(__file__).resolve().parent.parent / "calibration" / "key_numbers.json"


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def load_key_number_weights(path: Optional[Path] = None) -> Dict[int, float]:
    path = path or KEY_NUMBERS_PATH
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            return {int(k): float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    return dict(DEFAULT_KEY_NUMBER_WEIGHTS)


class MarginDistribution:
    """Discrete distribution over integer margins, key-number aware."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        key_weights: Optional[Dict[int, float]] = None,
        span: float = 5.0,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.key_weights = key_weights if key_weights is not None else load_key_number_weights()

        lo = int(math.floor(mu - span * sigma))
        hi = int(math.ceil(mu + span * sigma))
        # Margins of 0 don't exist in college football (overtime settles ties).
        margins = [m for m in range(lo, hi + 1) if m != 0]
        raw = {}
        for m in margins:
            density = normal_pdf((m - self.mu) / self.sigma)
            raw[m] = density * self.key_weights.get(abs(m), 1.0)
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("degenerate margin distribution")
        self.pmf: Dict[int, float] = {m: p / total for m, p in raw.items()}

    def p_push(self, line: float) -> float:
        """Probability the home margin lands exactly on ``line``."""
        if abs(line - round(line)) > 1e-9:
            return 0.0
        return self.pmf.get(int(round(line)), 0.0)

    def p_home_cover(self, line: float) -> float:
        """P(margin > line): the home side beats that number outright."""
        return sum(p for m, p in self.pmf.items() if m > line)

    def p_away_cover(self, line: float) -> float:
        return sum(p for m, p in self.pmf.items() if m < line)

    def p_home_win(self) -> float:
        return self.p_home_cover(0.0)

    def mean(self) -> float:
        return sum(m * p for m, p in self.pmf.items())


# -- pricing -----------------------------------------------------------------
def american_to_decimal(price: int) -> float:
    if price == 0:
        raise ValueError("price cannot be zero")
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def american_to_payout(price: int) -> float:
    """Profit per 1 unit risked."""
    return american_to_decimal(price) - 1.0


def implied_probability(price: int) -> float:
    return 1.0 / american_to_decimal(price)


def remove_vig(price_a: int, price_b: int) -> tuple[float, float]:
    """Two-way no-vig probabilities."""
    ia, ib = implied_probability(price_a), implied_probability(price_b)
    total = ia + ib
    if total <= 0:
        raise ValueError("invalid prices")
    return ia / total, ib / total


def expected_value(p_win: float, price: int = -110, p_push: float = 0.0) -> float:
    """EV per 1 unit risked. Pushes return the stake, so they contribute 0."""
    payout = american_to_payout(price)
    p_lose = max(0.0, 1.0 - p_win - p_push)
    return p_win * payout - p_lose


def kelly_fraction(p_win: float, price: int = -110, p_push: float = 0.0) -> float:
    """Full-Kelly fraction of bankroll. Zero when there's no edge."""
    b = american_to_payout(price)
    if b <= 0:
        return 0.0
    p_lose = max(0.0, 1.0 - p_win - p_push)
    # Renormalize over non-push outcomes: a push is a no-action bet.
    live = p_win + p_lose
    if live <= 0:
        return 0.0
    p, q = p_win / live, p_lose / live
    f = (b * p - q) / b
    return max(0.0, f)


@dataclass
class BetEvaluation:
    side: str  # "home" | "away" | "none"
    line: float  # the market spread from the bet side's perspective
    edge_points: float
    cover_probability: float
    push_probability: float
    expected_value: float
    kelly: float
    units: float

    @property
    def is_play(self) -> bool:
        return self.side != "none"


def evaluate_spread_bet(
    projected_margin: float,
    spread: Optional[float],
    sigma: float,
    price: int = -110,
    kelly_multiplier: float = 0.25,
    bankroll_units: float = 100.0,
    min_edge_points: float = 1.5,
    max_units: float = 3.0,
    key_weights: Optional[Dict[int, float]] = None,
    edge_shrink: float = 0.35,
) -> BetEvaluation:
    """Compare a projected margin against the market spread.

    ``spread`` is the home-side spread (-7 = home favored by 7). Returns the
    side with positive edge, or side ``"none"`` when the edge is too small to
    bet or no line is available.

    About ``edge_shrink``. Taking the raw disagreement at face value assumes our
    number is correct and the market is simply wrong by that many points. It
    isn't: the closing line is the sharpest public estimate there is, and when a
    model built from public ratings disagrees with it by seven points, most of
    that gap is our error, not theirs. Only a fraction of the disagreement is
    real signal, so the edge is shrunk toward the market before it is turned
    into a probability or a stake.

    The raw edge is still reported — that's the honest description of the
    disagreement — but the money is sized off the shrunk one.
    ``cfbmeta backtest --fit`` estimates the right fraction by regressing actual
    margins on the market line and our disagreement with it.
    """
    if spread is None:
        return BetEvaluation("none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    market_margin = -float(spread)
    edge = projected_margin - market_margin

    shrink = max(0.0, min(1.0, edge_shrink))
    settled_margin = market_margin + edge * shrink
    dist = MarginDistribution(settled_margin, sigma, key_weights=key_weights)
    push = dist.p_push(market_margin)

    if edge >= 0:
        side, p_cover = "home", dist.p_home_cover(market_margin)
        side_line = float(spread)
    else:
        side, p_cover = "away", dist.p_away_cover(market_margin)
        side_line = -float(spread)

    ev = expected_value(p_cover, price, push)
    kelly = kelly_fraction(p_cover, price, push)
    units = min(kelly * kelly_multiplier * bankroll_units, max_units)

    if abs(edge) < min_edge_points or ev <= 0:
        return BetEvaluation("none", side_line, edge, p_cover, push, ev, kelly, 0.0)

    return BetEvaluation(side, side_line, edge, p_cover, push, ev, kelly, round(units, 2))


def evaluate_total_bet(
    projected_total: Optional[float],
    market_total: Optional[float],
    sigma: float,
    price: int = -110,
    kelly_multiplier: float = 0.25,
    bankroll_units: float = 100.0,
    min_edge_points: float = 2.5,
    max_units: float = 3.0,
    edge_shrink: float = 0.35,
) -> BetEvaluation:
    """Same idea for over/under. Totals need a bigger edge to be actionable."""
    if projected_total is None or market_total is None:
        return BetEvaluation("none", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    edge = projected_total - market_total
    # Same market-respect shrink as the spread: see evaluate_spread_bet.
    z = (edge * max(0.0, min(1.0, edge_shrink))) / sigma
    p_over = normal_cdf(z)
    side = "over" if edge >= 0 else "under"
    p_cover = p_over if edge >= 0 else 1.0 - p_over

    ev = expected_value(p_cover, price)
    kelly = kelly_fraction(p_cover, price)
    units = min(kelly * kelly_multiplier * bankroll_units, max_units)

    if abs(edge) < min_edge_points or ev <= 0:
        return BetEvaluation("none", float(market_total), edge, p_cover, 0.0, ev, kelly, 0.0)
    return BetEvaluation(side, float(market_total), edge, p_cover, 0.0, ev, kelly, round(units, 2))
