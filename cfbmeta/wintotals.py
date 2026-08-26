"""Season win totals: the softest market this model can reach.

Why this market and not game spreads
------------------------------------
The spread work reached a clear and unwelcome conclusion. Against closing
lines on individual games the model's errors correlate with the market's at
0.938 — it is 94% the same bet as the closing line — and the fitted edge is
0.096 with a 95% interval of -0.018 to +0.210, which contains zero. Three
rounds of accuracy improvement moved the against-the-spread record from 49.8%
to 50.4% against a break-even of 52.4%. There is no measurable edge there, and
fitting harder for one mostly fits noise.

Season win totals are a different animal:

* They are posted months out and repriced rarely, so the number is not the
  distillation of everything known at kickoff that a closing spread is.
* Limits are low and sharp attention is thin, which is exactly the condition
  under which a slow number survives.
* The object we produce — a full distribution over a team's win count — is
  precisely what this market prices, rather than a point estimate that has to
  be converted into one.

That last point matters more than it sounds. The simulation's win
distributions are *calibrated*: across 2023-25, stated 80% intervals contained
the real win total 79.4% of the time. That is a measured property, not a
claim, and it is the prerequisite for turning a distribution into a price.

What this does not establish
----------------------------
Calibration is not edge. It says the model's uncertainty is honestly sized; it
says nothing about whether the model's *centre* beats the number a book hung
in May. Without a history of posted win totals to test against, that cannot be
measured here, and so:

* ``shrink`` treats only a fraction of any disagreement with the number as
  real signal, exactly as ``edge_shrink`` does for spreads, and defaults
  conservatively.
* Every card this produces is labelled unvalidated, and stays labelled until
  a season of results says otherwise.

The lesson from the spread work is worth restating in this context: the
threshold matters more than the model. Betting every spread disagreement above
1.5 points lost 3.8% over 1,731 games. Do not set the win-total threshold low
because the numbers look tempting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from .probability import expected_value, implied_probability, kelly_fraction
from .ratings import normalize_team

log = logging.getLogger(__name__)

DEFAULT_LINES_PATH = Path(__file__).resolve().parent.parent / "win-totals.yml"

# Fraction of our disagreement with the posted number treated as real signal.
# Same reasoning as edge_shrink: 1.0 would assert the book is wrong by the full
# amount, which is not a defensible starting position.
DEFAULT_SHRINK = 0.35
# Below this much disagreement (after shrinking), not a play. Deliberately well
# clear of zero — see the module docstring on thresholds.
DEFAULT_MIN_EDGE_WINS = 0.75

# Teams the model is measurably wrong about, excluded from betting rather than
# corrected. Across 2023-25 it projected the service academies 1.62 wins too
# LOW on average (Navy +3.33, Army +5.5 in 2024 alone) against no detectable
# bias anywhere else — every conference's error interval contains zero. The
# mechanism is understood: option offenses post low play counts and almost no
# explosiveness, which the efficiency composite reads as bad football, and
# academy rosters carry recruiting ratings that miss their retention and
# experience edge entirely.
#
# Nine team-seasons is enough to know the model is wrong and nowhere near
# enough to fit a correction, so these are passed. The first card built here
# wanted unders on all three, which is betting straight into the bias.
DEFAULT_EXCLUSIONS = ("army", "navy", "air force")


@dataclass
class WinTotalLine:
    """A posted season win total for one team.

    The over and the under can sit at *different numbers*, because books
    disagree — 30 of 139 teams are posted a full win apart somewhere. An over
    bettor wants the lowest number on the board and an under bettor the
    highest, so each side carries its own total and its own price.
    """

    team: str
    total: float                       # the number for the over
    over_price: int = -110
    under_price: int = -110
    under_total: Optional[float] = None  # None means the same number
    book: str = ""
    under_book: str = ""
    under_price_estimated: bool = False

    @property
    def key(self) -> str:
        return normalize_team(self.team)

    @property
    def under_number(self) -> float:
        return self.total if self.under_total is None else self.under_total


@dataclass
class WinTotalBet:
    team: str
    display: str = ""
    conference: str = ""
    side: str = "none"          # "over" | "under" | "none"
    total: float = 0.0
    price: int = -110
    book: str = ""
    model_wins: float = 0.0
    market_wins: float = 0.0   # what the price says, not what the number says
    blended_wins: float = 0.0
    edge_wins: float = 0.0      # blended minus the posted number
    p_over: float = 0.0
    p_under: float = 0.0
    p_push: float = 0.0
    expected_value: float = 0.0
    kelly: float = 0.0
    units: float = 0.0
    price_estimated: bool = False
    notes: List[str] = field(default_factory=list)

    def price_needed(self, target_ev: float = 0.0) -> Optional[int]:
        """The American price at which this bet would clear ``target_ev``.

        Worth knowing whenever the quoted price is a guess: the scrape publishes
        only over prices, so an under is priced off an assumed hold. If the edge
        is real but the assumed price kills it, the useful output is not "pass"
        but "pass unless you can get better than this".
        """
        p, push = self.win_probability, self.p_push
        live = 1.0 - push
        if p <= 0 or live <= 0:
            return None
        p_lose = live - p
        # target = p*payout - p_lose  ->  payout = (target + p_lose) / p
        payout = (target_ev + p_lose) / p
        if payout <= 0:
            return None
        return (int(round(100 * payout)) if payout >= 1.0
                else int(round(-100 / payout)))

    @property
    def is_play(self) -> bool:
        return self.side != "none"

    @property
    def win_probability(self) -> float:
        return self.p_over if self.side == "over" else self.p_under


def smeared_cdf(win_counts: Dict[int, float], x: float) -> float:
    """P(wins <= x), treating each integer's mass as spread over +/- half a win.

    The win count is discrete but the shrink shifts it by a fractional amount,
    so *some* interpolation is unavoidable. Smearing each integer uniformly
    across its unit interval is the standard continuity correction, and unlike
    fitting a normal it keeps the distribution's real shape — which is skewed
    for teams pressed against 0 wins or a full schedule.
    """
    total = 0.0
    for wins, mass in win_counts.items():
        low = wins - 0.5
        fraction = (x - low)
        if fraction <= 0.0:
            continue
        total += mass * (1.0 if fraction >= 1.0 else fraction)
    return min(1.0, max(0.0, total))


def outcome_probabilities(
    win_counts: Dict[int, float], total: float, shift: float = 0.0
) -> Tuple[float, float, float]:
    """(over, under, push) for a posted total, after shifting by ``shift``.

    A half-win line cannot push. A whole-number line pushes on exactly that
    many wins, and that mass is real — around 12% of the distribution for a
    typical team — so it must not be quietly folded into one side.
    """
    if abs(total - round(total)) < 1e-9:
        whole = round(total)
        p_under = smeared_cdf(win_counts, whole - 0.5 - shift)
        p_over = 1.0 - smeared_cdf(win_counts, whole + 0.5 - shift)
        p_push = max(0.0, 1.0 - p_over - p_under)
    else:
        cut = total - shift
        p_under = smeared_cdf(win_counts, cut)
        p_over = 1.0 - p_under
        p_push = 0.0
    return p_over, p_under, p_push


def market_implied_wins(
    win_counts: Dict[int, float], total: float, over_price: int, under_price: int
) -> float:
    """What win total the market is actually quoting, once the juice is read.

    A posted number is not the market's opinion. A 7.5 with the over at -175 is
    a book saying a team wins about 8.2 games, not 7.5 — the disagreement lives
    in the price, and comparing a model mean to the bare number misses it
    entirely. Worse, it misreads a *heavily juiced* line as a fair one and
    reports enormous expected value on what is really a coin flip.

    So: de-vig the two prices into a fair P(over), then find the shift of our
    own distribution that reproduces it. That shift is the market's number.
    """
    p_over = implied_probability(over_price)
    p_under = implied_probability(under_price)
    if p_over + p_under <= 0:
        return total
    fair_over = p_over / (p_over + p_under)

    # The distribution is monotone in the shift, so bisect.
    low, high = -12.0, 12.0
    for _ in range(60):
        mid = (low + high) / 2.0
        got, _, _ = outcome_probabilities(win_counts, total, mid)
        if got < fair_over:
            low = mid
        else:
            high = mid
    mean = sum(w * p for w, p in win_counts.items())
    return mean + (low + high) / 2.0


def evaluate(
    outcome,
    line: WinTotalLine,
    shrink: float = DEFAULT_SHRINK,
    min_edge_wins: float = DEFAULT_MIN_EDGE_WINS,
    kelly_multiplier: float = 0.25,
    bankroll_units: float = 100.0,
    max_units: float = 3.0,
) -> WinTotalBet:
    """Price one posted total against the simulated win distribution."""
    bet = WinTotalBet(
        team=outcome.team, display=getattr(outcome, "display", outcome.team),
        conference=getattr(outcome, "conference", ""),
        total=line.total, book=line.book, model_wins=outcome.mean_wins,
    )
    if not outcome.win_counts:
        bet.notes.append("No simulated distribution for this team.")
        return bet

    # Each side is priced against its own number, since they can differ.
    def side_for(total: float, price: int, want_over: bool):
        # Disagree with what the market is really quoting, not with the
        # number printed next to the juice.
        other = line.under_price if want_over else line.over_price
        implied = market_implied_wins(
            outcome.win_counts, total,
            price if want_over else other, other if want_over else price,
        )
        raw = outcome.mean_wins - implied
        blended = implied + shrink * raw
        shift = blended - outcome.mean_wins
        p_over, p_under, p_push = outcome_probabilities(
            outcome.win_counts, total, shift
        )
        p = p_over if want_over else p_under
        return {
            "total": total, "price": price, "p": p, "p_push": p_push,
            "p_over": p_over, "p_under": p_under, "implied": implied,
            "blended": blended, "edge": blended - implied,
            "ev": expected_value(p, price, p_push),
        }

    over = side_for(line.total, line.over_price, True)
    under = side_for(line.under_number, line.under_price, False)
    chosen = over if over["ev"] >= under["ev"] else under
    side = "over" if chosen is over else "under"

    bet.total = chosen["total"]
    bet.market_wins = chosen["implied"]
    bet.blended_wins = chosen["blended"]
    bet.edge_wins = chosen["edge"]
    bet.p_over, bet.p_under, bet.p_push = (
        chosen["p_over"], chosen["p_under"], chosen["p_push"]
    )
    bet.expected_value = ev = chosen["ev"]
    bet.price = price = chosen["price"]
    p = chosen["p"]
    if side == "under":
        bet.book = line.under_book or line.book
        bet.price_estimated = line.under_price_estimated
    if abs(bet.edge_wins) < min_edge_wins:
        bet.notes.append(
            f"Edge {abs(bet.edge_wins):.2f} wins is under the {min_edge_wins:.2f} threshold."
        )
        return bet
    if ev <= 0:
        need = bet.price_needed()
        hint = f" Needs better than {need:+d}." if need is not None else ""
        estimated = " (that price is an estimate)" if bet.price_estimated else ""
        bet.notes.append(
            f"Priced out at {bet.price:+d}{estimated}.{hint}"
        )
        return bet
    # Direction has to agree with the edge, or we are betting the vig line
    # rather than the disagreement.
    if (side == "over") != (bet.edge_wins > 0):
        bet.notes.append("Best price sits opposite our lean; passing.")
        return bet

    bet.side = side
    bet.kelly = kelly_fraction(p, price, bet.p_push)
    bet.units = round(
        min(bet.kelly * kelly_multiplier * bankroll_units, max_units), 2
    )
    return bet


def load_lines(path: Optional[Path] = None) -> List[WinTotalLine]:
    """Read posted totals from win-totals.yml.

    Manual entry on purpose. Season win totals are not in CFBD, and the odds
    feed only carries them on paid plans — but they are trivially readable off
    any book, they move rarely, and typing them in once a year is a smaller
    cost than an integration that silently goes stale.
    """
    path = Path(path) if path else DEFAULT_LINES_PATH
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping of team -> total")

    lines: List[WinTotalLine] = []
    for team, value in (raw.get("totals") or {}).items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            lines.append(WinTotalLine(team=str(team), total=float(value)))
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{team}: expected a number or a mapping, got {value!r}")
        lines.append(WinTotalLine(
            team=str(team),
            total=float(value["total"]),
            over_price=int(value.get("over", -110)),
            under_price=int(value.get("under", -110)),
            under_total=(float(value["under_total"])
                         if value.get("under_total") is not None else None),
            book=str(value.get("book", raw.get("book", "")) or ""),
        ))
    log.info("loaded %d posted win totals from %s", len(lines), path.name)
    return lines


def build_card(
    simulation,
    lines: Iterable[WinTotalLine],
    shrink: float = DEFAULT_SHRINK,
    min_edge_wins: float = DEFAULT_MIN_EDGE_WINS,
    kelly_multiplier: float = 0.25,
    bankroll_units: float = 100.0,
    max_units: float = 3.0,
    max_total_units: Optional[float] = None,
    exclude: Optional[Iterable[str]] = DEFAULT_EXCLUSIONS,
) -> List[WinTotalBet]:
    """Price every posted total, best plays first.

    Win totals on one model are correlated the same way a slate of spreads is —
    they share the ratings, and a season that goes badly for the model goes
    badly across the whole card at once. ``max_total_units`` scales the card
    down proportionally rather than pretending otherwise.
    """
    excluded = {normalize_team(t) for t in (exclude or ())}
    bets: List[WinTotalBet] = []
    for line in lines:
        outcome = simulation.teams.get(line.key)
        if outcome is None:
            log.debug("no simulated team for posted total %r", line.team)
            continue
        if line.key in excluded:
            skipped = WinTotalBet(
                team=line.key, display=getattr(outcome, "display", line.team),
                conference=getattr(outcome, "conference", ""),
                total=line.total, model_wins=outcome.mean_wins,
            )
            skipped.notes.append(
                "Excluded: the model has a measured bias on this team."
            )
            bets.append(skipped)
            continue
        bets.append(evaluate(
            outcome, line, shrink=shrink, min_edge_wins=min_edge_wins,
            kelly_multiplier=kelly_multiplier, bankroll_units=bankroll_units,
            max_units=max_units,
        ))

    plays = [b for b in bets if b.is_play]
    staked = sum(b.units for b in plays)
    if max_total_units and staked > max_total_units:
        scale = max_total_units / staked
        for bet in plays:
            bet.units = round(bet.units * scale, 2)
        log.info(
            "win-total card scaled %.1fu -> %.1fu for total exposure",
            staked, max_total_units,
        )

    bets.sort(key=lambda b: (not b.is_play, -abs(b.expected_value)))
    return bets
