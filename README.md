[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hyrenk722/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hyrenk722/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Earners over Players — GTA VI Trophy Rarity Forecaster

Entry for the make.fun competition **gta6-trophy-rarity-forecast-01**:
"Forecast how many GTA VI players earn each trophy in week one".

## What it does

`main.py` reads one case file per run — every read the house recorded of
the game's Exophase trophy page before the case's checkpoint (players
tracked, and each trophy's earn rate) — and forecasts, for each of the
case's trophies, five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of its
earn rate at the case's target. Scoring is mean pinball loss on the logit
scale, so the quantiles are calibrated for that loss, not for a point
forecast.

## Method

An earn rate is a ratio — a trophy's earners over the players tracked —
and a launch week has two regimes: the tracked crowd floods in (diluting
every rate), then the crowd plays (earners catch up). The program
therefore fits the two sides separately and divides at the target:

- The read history is collapsed to distinct page states (Exophase
  refreshes on its own clock; a repeated page is one observation,
  weighted by how long it stood).
- Players tracked and each trophy's earners (rate × players / 100) get
  weighted least-squares lines against ln(1 + hours since release / 6 h),
  fit on the last 24 h and last 48 h windows; each trophy's slope is
  shrunk toward the cross-trophy mean slope by its own precision.
- A direct line of the trophy's logit rate is fit the same way, and the
  two views are averaged in logit space.
- Extrapolated increments are damped — hardest for the players curve
  (a launch crowd's growth saturates), less for earners, not at all for
  the direct view, whose slope is already the net of the two — capped,
  and anchored to each trophy's recent observed level; the median's move
  from the last observed rate is bounded, since no rate runs to 0% or
  100% in a week.
- The spread is analytic: each trophy's recent-window residual scatter,
  its slope's standard error projected over the extrapolation distance,
  and the disagreement between windows and views, combined in
  quadrature — the uncertainty the fits themselves report.
- A history that is a single frozen page (the launch-night practice
  shape) takes a stale-page branch: a page frozen since before launch is
  expected to refresh before the target, adding players far faster than
  earners, so rates dilute by the documented refresh multiplier while a
  trophy every player holds stays held by all.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

- Input: every `*.in` file in `MAKEFUN_INPUT_DIR` (default `/input`),
  one JSON case per file.
- Output: one `<case>.out` JSON file per case in `MAKEFUN_OUTPUT_DIR`
  (default `/output`), with the five quantiles for each trophy.
- The program is deterministic: the same input always produces
  byte-identical output. Python 3.13, standard library only.

## Repository layout

- `main.py` — the forecaster (the entry program).
- `LICENSE` — MIT license.
- `.makefun/owner.json` — entrant claim file naming the payout wallet
  for this entry.
