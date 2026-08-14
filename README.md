# cfb

College football meta-forecast. Distills the public rating systems into one
projected margin per game, grades it against the market, and produces the
readout every Thursday at 7am. (Email delivery is held for now — see setup.)

Same idea as running KenPom for basketball: don't build a rating from scratch,
distill the good ones and be disciplined about the adjustments and the sizing.

---

## Setup

### 1. Get a CFBD API key

Free at <https://collegefootballdata.com/key>. One key covers every rating
source used here — SP+, FPI, Elo, SRS, talent, lines, schedules, venues and
coaching records all come from that one API.

### 2. Get a Gmail app password

<https://myaccount.google.com/apppasswords>. A normal account password will be
rejected by SMTP; it has to be an app password.

### 3. Add repository secrets

**Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value |
| --- | --- |
| `CFBD_API_KEY` | your collegefootballdata.com key |
| `SMTP_USER` | your Gmail address |
| `SMTP_PASSWORD` | the app password from step 2 |
| `EMAIL_TO` | where the readout goes (comma-separated for several) |

Optional repository *variables* (not secrets): `SMTP_HOST`, `SMTP_PORT` if
you'd rather not use Gmail.

Never paste a key into a commit, an issue, or a chat window. If one does leak,
rotate it — CFBD keys are free to reissue and a Gmail app password can be
revoked from your Google account without touching your real password.

> **Email delivery is currently held.** The weekly workflow builds the full
> slate and uploads it as a downloadable artifact, but sends nothing. To turn
> sending on, add a repository **variable** named `SEND_EMAIL` set to `true`.
> Nothing else needs to change.

### 4. Confirm the plumbing

Run the **Checks** workflow manually with *"Also probe the live CFBD API"*
ticked. That runs `doctor`, which hits every endpoint, prints the fields each
one returned, and finishes by projecting a real slate end to end. If CFBD ever
renames a field or moves an endpoint, this is what tells you.

Locally, the same thing:

```bash
pip install -r requirements.txt
echo 'CFBD_API_KEY=your-key-here' > .env   # untracked; never commit it
python -m cfbmeta doctor
```

`.env` is read automatically and is gitignored, so the key lives in one local
file instead of being exported in every shell. Real environment variables
always win, so CI secrets are never overridden by a stray local file.

That's it. Once `SEND_EMAIL` is switched on, the forecast arrives every
Thursday at 7am Eastern. Until then it's waiting for you as a workflow
artifact each week.

---

## Usage

```bash
python -m cfbmeta run --no-email        # print this week's slate
python -m cfbmeta preview --week 0      # week 0 specifically
python -m cfbmeta preview --week 1
python -m cfbmeta preview --week 6      # render to preview.html
python -m cfbmeta run --dry-run         # full run, stops short of sending
python -m cfbmeta doctor                # verify credentials and endpoints
python -m cfbmeta backtest --years 5 --fit --write   # measure, then tune
```

---

## How the number is built

For each game:

1. **Every rating source proposes a neutral-field margin.** SP+, FPI and SRS
   are already published as net points per game, so they're used directly. Elo
   and the talent composite are on their own scales, so they're z-scored across
   FBS and stretched to match the spread of the SP+ ratings that season — which
   means the conversion recalibrates itself each year instead of relying on a
   hardcoded constant.
2. **Those are blended by weight.** Any source missing a rating for either team
   drops out and the weights renormalize over what's left, so a single API
   hiccup degrades the forecast instead of killing it.
3. **Home field is added once, on top** — not per source, since every source is
   quoted as a neutral-field rating.
4. **Situational adjustments** are added: coaching, rest, travel.
5. **The result is compared against the market** to produce an edge.

Early in the season, Elo and SRS describe a sample of one or two games. Their
weight is shifted toward the preseason-anchored sources (SP+, FPI, talent) and
returned week by week, controlled by `prior_weight_by_week`.

### Home field is per venue, not a constant

Every team is its own control:

```
raw_hfa(team) = (mean margin at home - mean margin away) / 2
```

Team quality cancels out of the difference, leaving mostly venue effect and
noise. Because it's noisy, it's shrunk hard toward the league mean with a
60-game pseudo-count, so two flukey home blowouts don't buy a team a six-point
home field. Altitude is handled separately as a *differential* against where
the visitor normally plays — Wyoming hosting Florida is a real edge, Wyoming
hosting Air Force is not.

### Coaching, measured rather than asserted

Fit how much SP+ rating a team's recruiting talent buys on average, then credit
the coach with the residual — how much better their teams finish than their raw
material predicts. Weighted toward recent seasons, shrunk by career length,
capped at ±1.5 points, plus a penalty for a first-year hire. It's a tiebreaker
on close games, never a driver.

### Uncertainty, and why the edges look small

Margins are modelled with σ ≈ 16 points, and college football margins are not
smoothly normal — they pile up on 3 and 7. A plain normal misprices any bet
sitting on a key number, so the margin distribution is discretized and those
integers are reweighted from the empirical distribution.

**The most important knob is `edge_shrink`.** Taking a raw disagreement at face
value assumes our number is right and the closing line is wrong by that full
amount. It isn't. The closing line is the sharpest public estimate available,
and when a model built from public ratings disagrees with it by seven points,
most of that gap is our error, not theirs. So only a fraction of the
disagreement is treated as real signal when sizing.

The raw edge is still reported — that's the honest description of the
disagreement — but the money is sized off the shrunk one. Without this, every
game with a five-point disagreement reads as a maximum bet, which is how these
systems talk people into ruin. `backtest --fit` estimates the right fraction by
regressing actual margins on the market line and on our disagreement with it.

Stakes are quarter-Kelly against a 100-unit bankroll, capped at 3 units.

---

## Preseason data, and what's actually available in August

The season is inferred from the date (`season: 0`), so an August 2026 run
requests **2026** everywhere — ratings, schedule, lines, coaches — and January
bowls still resolve to the 2026 season rather than 2027.

But requesting 2026 is not the same as getting it. In August:

| Source | Preseason availability |
| --- | --- |
| SP+ | published in spring, available |
| Talent composite | available once signing day closes |
| FPI | ESPN posts it closer to week 1 |
| SRS | derived from results — nothing until games are played |
| Elo | preseason values are carryover, often flat |

The dangerous case is not a missing source. It's a source that returns a row
for **every team with identical values** — a preseason SRS where everyone sits
at 0.0. That isn't a weak opinion, it's no opinion, and blending it silently
drags every projection toward pick'em while the run looks perfectly healthy.

So any source with fewer than 10 rated teams, or less than half a point of
spread between them, is dropped outright and the weights renormalize over what
remains. Every source's status is recorded and reported — in `doctor`, in the
run log, and in the footer of the email itself:

```
OK    SP+ 2026: 136 teams (spread 14.2)
OK    Talent 2026: 136 teams (spread 10.5)
WARN  SRS 2026: unusable — all 136 teams within 0.00 pts — no games played yet
WARN  FPI 2026: unusable — no data returned
```

A preseason forecast therefore leans on SP+ and talent, which is the honest
answer for August, and the readout says so on its face rather than implying
five sources agreed. If *no* source is usable the run fails loudly with the
audit attached, instead of emitting a slate of 0.0 margins.

This dovetails with the early-season weighting: week 0 and week 1 already shift
80% and 75% of the Elo/SRS weight to the preseason-anchored sources, so losing
them entirely costs less than it sounds.

---

## Tuning it with real results

```bash
python -m cfbmeta backtest --years 5 --fit --write
```

This fits the blend weights, σ, and the edge shrink against completed games and
writes them to `config.yml`.

**One caveat that matters.** CFBD serves *end-of-season* SP+, FPI and SRS
ratings for a given year. Using those to "predict" that same year's games is
lookahead — the ratings already know how the games turned out. So the default
basis (`--basis prior`) rates every game using the **previous** season's final
ratings, which is information that genuinely existed before kickoff. It
understates in-season accuracy, but it never flatters the model.

`--basis same` is available for comparing the sources against each other, since
the contamination affects them all similarly. Its error figures are not real,
they're labelled as such in the output, and `--write` refuses to save anything
derived from them.

---

## The Thursday 7am schedule

(Held for now — see the note in setup. The schedule still runs and still
produces the readout; it just doesn't mail it.)

GitHub Actions cron only speaks UTC, and 7am Eastern is 11:00 UTC in September
but 12:00 UTC after the November clock change. The workflow fires at **both**
hours every Thursday, and `run --check-time` exits immediately unless the local
hour actually matches. Exactly one of the two runs sends, all season, with no
schedule edit at the DST boundary. There's a test that walks 20 consecutive
Thursdays across the change and asserts exactly one send each week.

To move it: change `timezone` and `send_hour_local` in `config.yml`, and adjust
the two cron hours in `.github/workflows/weekly-forecast.yml` to bracket your
local send time.

Out of season, `resolve_week` finds no upcoming games and the run exits quietly
without mailing anything.

---

## Layout

```
cfbmeta/
  config.py         settings, YAML + env overrides
  sources/cfbd.py   API client: retries, path fallback, tolerant field access
  sources/market.py consolidates sportsbook lines into a median consensus
  ratings.py        normalizes every source onto the points scale
  hfa.py            per-venue home field, with shrinkage and altitude
  coaching.py       talent-adjusted coach residuals
  adjustments.py    rest, travel, short weeks
  model.py          the blend and the per-game projection
  probability.py    key numbers, cover probability, EV, Kelly
  backtest.py       accuracy measurement and weight fitting
  report.py         HTML and plain-text rendering
  email_send.py     SMTP delivery
  cli.py            run / preview / doctor / backtest
tests/              193 tests, no network required
```

### A note on the API client

The client is written defensively on purpose. CFBD has shipped both snake_case
and camelCase payloads across API versions and has moved endpoints between path
prefixes. Rather than betting on one shape, field lookups try several spellings
and each endpoint carries a list of candidate paths, using the first that
answers. `doctor` is how you confirm what the live API is actually returning.

---

## Weekly workflow

1. Thursday 7am — readout arrives, plays sorted by edge
2. Scan Kalshi positions, SWOT each
3. Apply JDK adjustment to the shortlist
4. Record best bets, budget units
5. Place selections
6. Record all wagers and outcomes

Feeding step 6 back into `backtest` is what makes the numbers get better.
