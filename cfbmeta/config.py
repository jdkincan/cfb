"""Configuration for the meta-forecast.

Values load from ``config.yml`` at the repo root and can be overridden by
environment variables (env wins, so CI can tweak a run without a commit).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yml"


@dataclass
class Weights:
    """Blend weights over component margins.

    These are priors. ``cfbmeta backtest --fit`` re-estimates them by least
    squares on completed games and writes the result back to config.yml.

    Market weight defaults to 0: the market is the benchmark we are trying to
    beat, so folding it into the forecast would launder the edge away. Set it
    above 0 only if a backtest says the blend is genuinely worse than the line.
    """

    sp_plus: float = 0.40
    fpi: float = 0.25
    elo: float = 0.15
    srs: float = 0.10
    talent: float = 0.10
    market: float = 0.00

    def normalized(self) -> Dict[str, float]:
        raw = asdict(self)
        model_keys = [k for k in raw if k != "market"]
        model_total = sum(raw[k] for k in model_keys)
        if model_total <= 0:
            raise ValueError("model component weights must sum to more than zero")
        # Model components share (1 - market weight); market keeps its own slice.
        share = 1.0 - raw["market"]
        out = {k: raw[k] / model_total * share for k in model_keys}
        out["market"] = raw["market"]
        return out


@dataclass
class Config:
    # --- season / scheduling -------------------------------------------------
    season: int = 0  # 0 means "infer from today's date"
    timezone: str = "America/New_York"
    send_hour_local: int = 7
    season_type: str = "regular"

    # --- blending ------------------------------------------------------------
    weights: Weights = field(default_factory=Weights)

    # Weeks 1-4 ratings are noisy, so early-season projections get pulled toward
    # a preseason prior (prior-year rating + talent + returning production).
    prior_weight_by_week: Dict[int, float] = field(
        default_factory=lambda: {1: 0.75, 2: 0.60, 3: 0.45, 4: 0.30, 5: 0.18, 6: 0.10, 7: 0.05}
    )

    # --- uncertainty ---------------------------------------------------------
    # SD of (actual margin - projected margin) for a decent CFB model. ~16 is
    # the well-known figure; backtest --fit re-estimates it from real results.
    sigma_margin: float = 16.0
    sigma_total: float = 13.5

    # --- home field ----------------------------------------------------------
    league_hfa: float = 2.35
    hfa_shrink_games: float = 60.0  # pseudo-count for shrinking venue HFA to league mean
    hfa_max: float = 5.0
    hfa_min: float = 0.0
    altitude_bonus_per_1k_ft: float = 0.22
    altitude_threshold_ft: float = 3000.0

    # --- adjustment caps (keep any single knob from dominating) --------------
    coach_adj_cap: float = 1.5
    rest_adj_cap: float = 2.0
    travel_adj_cap: float = 1.2
    total_adj_cap: float = 4.0

    # --- rest / travel -------------------------------------------------------
    bye_week_bonus: float = 1.0
    short_week_penalty: float = 0.8
    rest_day_value: float = 0.10  # points per day of rest differential
    travel_penalty_per_1k_mi: float = 0.35

    # --- betting -------------------------------------------------------------
    vig_price: int = -110
    # Fraction of our disagreement with the market treated as real signal when
    # sizing a bet. 1.0 would mean "our number is right and the closing line is
    # wrong by the full amount", which is not a defensible starting assumption
    # for a model built from public ratings. Estimate it with
    # `backtest --fit`; see evaluate_spread_bet for the reasoning.
    edge_shrink: float = 0.35
    kelly_fraction: float = 0.25
    bankroll_units: float = 100.0
    min_edge_points: float = 1.5  # below this, don't call it a play
    strong_edge_points: float = 3.0
    max_units_per_play: float = 3.0

    # --- reporting -----------------------------------------------------------
    include_non_fbs: bool = False
    email_subject_prefix: str = "CFB Meta Forecast"

    # --- data ----------------------------------------------------------------
    cfbd_base_url: str = "https://api.collegefootballdata.com"
    request_timeout: int = 30
    max_retries: int = 4
    cache_dir: str = "data/cache"
    cache_ttl_minutes: int = 360

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        data: Dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            data = loaded

        weights_data = data.pop("weights", {}) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "weights"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

        cfg = cls(**data)
        if weights_data:
            bad = set(weights_data) - set(Weights.__dataclass_fields__)
            if bad:
                raise ValueError(f"unknown weight keys: {sorted(bad)}")
            cfg.weights = Weights(**weights_data)

        # prior_weight_by_week keys arrive as ints from YAML but be forgiving.
        cfg.prior_weight_by_week = {int(k): float(v) for k, v in cfg.prior_weight_by_week.items()}
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        if os.getenv("CFB_SEASON"):
            self.season = int(os.environ["CFB_SEASON"])
        if os.getenv("CFB_TIMEZONE"):
            self.timezone = os.environ["CFB_TIMEZONE"]
        if os.getenv("CFB_SIGMA"):
            self.sigma_margin = float(os.environ["CFB_SIGMA"])
        if os.getenv("CFB_MIN_EDGE"):
            self.min_edge_points = float(os.environ["CFB_MIN_EDGE"])

    def resolved_season(self, today=None) -> int:
        if self.season:
            return self.season
        import datetime as _dt

        today = today or _dt.date.today()
        # A season labelled YYYY runs Aug YYYY through early Jan YYYY+1.
        return today.year if today.month >= 3 else today.year - 1

    def prior_weight(self, week: int) -> float:
        return float(self.prior_weight_by_week.get(int(week), 0.0))

    def save(self, path: Path | str | None = None) -> None:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        data = asdict(self)
        data["weights"] = asdict(self.weights)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
