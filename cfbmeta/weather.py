"""Weather adjustments, gated behind CFBD's Tier 1 subscription.

``/games/weather`` answers 401 on the free key with an explicit message:
"This endpoint requires a Patreon subscription at Tier 1 or higher." Tier 1 is
$1/month, so this is wired and ready rather than deferred — it activates the
moment the key is upgraded and degrades silently to zero adjustment until then.

What weather is actually worth, and to what
-------------------------------------------
Wind is the dominant effect and it lands almost entirely on **totals**, not
spreads. Passing efficiency and kicking both fall off above roughly 15 mph, and
hard above 20. Precipitation matters less than people assume — teams adapt, and
a wet field slows both offences. Cold is the weakest of the three and is mostly
a proxy for wind anyway.

The spread effect exists but is small and asymmetric: bad weather compresses
margins, which helps the underdog. A pass-first team loses more than a
run-first team, but the model has no play-style split, so the compression is
applied symmetrically toward the dog and kept deliberately tiny.

Cold is handled as a *differential*, not a level. A December kickoff at 20°F is
ordinary for Wisconsin and genuinely disruptive for a team that has played its
whole season in Florida — so the penalty scales with how far the visitor is
from its own normal, using latitude as a crude proxy for climate. Applying a
flat cold penalty to both teams would be worse than applying none.

Domes are excluded outright: an indoor game has no weather regardless of what
the forecast says about the parking lot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from .ratings import normalize_team
from .sources.cfbd import pick, pick_float

log = logging.getLogger(__name__)

# Wind above this many mph starts to matter; effect grows from there.
WIND_THRESHOLD_MPH = 12.0
# Points off the total per mph above the threshold.
TOTAL_PER_MPH = 0.45
# Points off the total for measurable precipitation.
PRECIP_TOTAL_PENALTY = 1.8
# Spread compression toward the underdog, per mph of excess wind. Deliberately
# an order of magnitude smaller than the total effect.
SPREAD_COMPRESSION_PER_MPH = 0.035
MAX_SPREAD_EFFECT = 1.5
MAX_TOTAL_EFFECT = 9.0
# Below this temperature, a team unused to cold starts to be affected.
COLD_THRESHOLD_F = 32.0


@dataclass
class GameWeather:
    game_id: str
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    humidity: Optional[float] = None
    condition: str = ""
    dome: bool = False

    @property
    def has_data(self) -> bool:
        return not self.dome and any(
            v is not None for v in (self.temperature, self.wind_speed, self.precipitation)
        )

    @property
    def excess_wind(self) -> float:
        if self.dome or self.wind_speed is None:
            return 0.0
        return max(0.0, self.wind_speed - WIND_THRESHOLD_MPH)

    def total_effect(self) -> float:
        """Points to subtract from a projected total. Never positive."""
        if self.dome:
            return 0.0
        effect = self.excess_wind * TOTAL_PER_MPH
        if self.precipitation:
            effect += PRECIP_TOTAL_PENALTY
        return -round(min(effect, MAX_TOTAL_EFFECT), 2)

    def describe(self) -> str:
        if self.dome:
            return "dome"
        parts = []
        if self.temperature is not None:
            parts.append(f"{self.temperature:.0f}°F")
        if self.wind_speed is not None:
            parts.append(f"{self.wind_speed:.0f} mph wind")
        if self.precipitation:
            parts.append("precipitation")
        if self.condition:
            parts.append(self.condition)
        return ", ".join(parts) or "no data"


@dataclass
class WeatherModel:
    games: Dict[str, GameWeather] = field(default_factory=dict)
    available: bool = False
    reason: str = ""
    # team key -> latitude, used to judge who is unaccustomed to cold.
    latitudes: Dict[str, float] = field(default_factory=dict)

    def for_game(
        self, game_id: Any, home_team: str, away_team: str, favorite_is_home: bool
    ) -> Dict[str, Any]:
        """Spread and total adjustments for one game, from the home side."""
        blank = {"total": 0.0, "spread": 0.0, "summary": "", "available": False}
        entry = self.games.get(str(game_id))
        if entry is None or not entry.has_data:
            return blank

        # Wind compresses margins, which helps whoever is getting points.
        compression = min(
            entry.excess_wind * SPREAD_COMPRESSION_PER_MPH, MAX_SPREAD_EFFECT
        )
        spread = -compression if favorite_is_home else compression

        # Cold hurts the side less used to it, scaled by how unusual it is for
        # them rather than by the raw temperature.
        if entry.temperature is not None and entry.temperature < COLD_THRESHOLD_F:
            home_lat = self.latitudes.get(normalize_team(home_team))
            away_lat = self.latitudes.get(normalize_team(away_team))
            if home_lat is not None and away_lat is not None:
                # Positive when the visitor comes from a warmer climate.
                gap = home_lat - away_lat
                if abs(gap) > 4.0:
                    severity = (COLD_THRESHOLD_F - entry.temperature) / 30.0
                    spread += max(-0.8, min(0.8, gap * 0.06 * severity))

        return {
            "total": entry.total_effect(),
            "spread": round(max(-MAX_SPREAD_EFFECT, min(MAX_SPREAD_EFFECT, spread)), 2),
            "summary": entry.describe(),
            "available": True,
        }


def parse_rows(rows: Iterable[dict]) -> Dict[str, GameWeather]:
    out: Dict[str, GameWeather] = {}
    for row in rows or []:
        gid = pick(row, "id", "gameId", "game_id")
        if gid is None:
            continue
        venue = (pick(row, "venue", default="") or "")
        dome = bool(pick(row, "dome", default=False)) or "dome" in str(venue).lower()
        out[str(gid)] = GameWeather(
            game_id=str(gid),
            temperature=pick_float(row, "temperature", "temp"),
            wind_speed=pick_float(row, "windSpeed", "wind_speed"),
            precipitation=pick_float(row, "precipitation", "precip"),
            humidity=pick_float(row, "humidity"),
            condition=str(pick(row, "weatherCondition", "condition", default="") or ""),
            dome=dome,
        )
    return out


def build_weather_model(
    client, season: int, week: int, teams: Optional[Iterable[dict]] = None
) -> WeatherModel:
    """Load weather for a week, or report why it is unavailable."""
    model = WeatherModel()
    try:
        rows = client.get("weather", year=season, week=week)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "Tier" in message or "401" in message:
            model.reason = (
                "weather needs a CFBD Patreon subscription at Tier 1 ($1/month); "
                "running without it"
            )
        else:
            model.reason = f"weather unavailable: {message[:120]}"
        log.info(model.reason)
        return model

    model.games = parse_rows(rows)
    model.available = bool(model.games)

    for team in teams or []:
        name = pick(team, "school", "team")
        lat = pick_float(team, "location.latitude", "latitude")
        if name and lat is not None:
            model.latitudes[normalize_team(name)] = lat

    log.info("weather loaded for %d games", len(model.games))
    return model
