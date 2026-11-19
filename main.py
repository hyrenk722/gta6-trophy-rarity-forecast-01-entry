"""gta6-trophy-rarity-forecast-01 entry: a windowed earners/players forecaster.

A trophy's earn rate is a ratio -- its earners over the players Exophase
tracks -- and a launch week has two regimes: the tracked crowd floods in
(diluting every rate) and then the crowd plays (earners catch up). A line
fit over the whole history averages the two regimes into mush, so this
program fits short log-time lines on recent windows and damps what it
extrapolates:

  - two views of every trophy: its earners (rate x players / 100 at each
    distinct page state) against the players curve, and its logit rate
    directly; the views are averaged in logit space;
  - each view is a weighted least-squares line of the log quantity against
    ln(1 + hours since release / 6 h), fit on the last 24 h and the last
    48 h of page states and ensembled by median, its slope shrunk toward
    the cross-trophy mean slope by the fit's own precision;
  - the extrapolated increment past the last state is damped -- hardest for
    the players curve (a launch crowd's growth saturates), less for
    earners (the crowd already tracked keeps playing), not at all for the
    direct rate view, whose slope is already the net of the two;
  - every fit is anchored to the trophy's recent observed level, so a fit
    that drifts off the last page is pulled back to it.

The history is first collapsed to distinct page states: Exophase refreshes
on its own clock, most reads repeat the previous page, and a repeated page
is one observation, weighted by how long it stood. The spread is analytic:
each trophy's recent-window residual scatter and its slope's standard
error projected over the extrapolation distance, combined in quadrature
with a floor -- the uncertainty the fits themselves report, in the current
regime, rather than a constant widened by a regime the case has left. A
case whose history is a single frozen page (the launch-night
practice shape) takes the stale-page branch: a page frozen since before
launch is expected to refresh before the target, adding players far faster
than earners, so rates dilute by the documented refresh multiplier while a
trophy every player holds stays held by all.

Standard library only, no network, no randomness: two runs of a case write
the same bytes. Run as the grader runs it, `python3 -P -s /bot/main.py`
with the case's file alone in the input directory; MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output outside the container.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_KNOWN = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()
TAU = 6.0
WINDOWS = (24.0, 48.0)
MAX_STATES = 240
ANCHOR = 0.9
DAMP_PLAYERS = 0.7
DAMP_EARNERS = 0.65
W_EARNERS_VIEW = 0.4
SIGMA_CAL = 1.0
SIGMA_MIN = 0.18
SIGMA_MAX = 1.9


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 700 else 100.0
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def normsinv(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_for(level: float) -> float:
    if level in Z_KNOWN:
        return Z_KNOWN[level]
    return normsinv(level)


def parse_time(s: str) -> float:
    """ISO 8601 UTC string to hours since the release anchor."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.timestamp() - RELEASE_EPOCH) / 3600.0


def x_of(t_hours: float) -> float:
    return math.log1p(max(t_hours, 0.0) / TAU)


class Fit:
    __slots__ = ("a", "b", "resid", "se_b", "xbar")

    def __init__(self, a, b, resid, se_b, xbar):
        self.a, self.b, self.resid, self.se_b, self.xbar = a, b, resid, se_b, xbar


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns a Fit."""
    sw = sum(ws)
    if sw <= 0 or not xs:
        return Fit(ys[0] if ys else 0.0, 0.0, 1.0, 1e9, xs[0] if xs else 0.0)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx < 1e-12:
        return Fit(ybar, 0.0, 1.0, 1e9, xbar)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    # Degrees of freedom from the point count, not the weight sum: duration
    # weights are not replicate observations.
    n = len(xs)
    resid = math.sqrt((sse / sw) * (n / max(1, n - 2)))
    return Fit(a, b, resid, resid / math.sqrt(sxx), xbar)


def shrink_slopes(fits):
    """Shrink fits' slopes toward their precision-weighted mean slope."""
    bs = [f.b for f in fits]
    # The standard-error floor matters: a fit on a constant series (a trophy
    # flat at 0.00%) reports se 0 and would otherwise take infinite precision
    # weight, switching shrinkage off exactly where borrowing strength is
    # needed most. The floor is the rounding noise of the printed page.
    ses = [max(f.se_b, 0.08) for f in fits]
    wsum = sum(1.0 / (s * s) for s in ses)
    bbar = sum(b / (s * s) for b, s in zip(bs, ses)) / wsum if wsum > 0 else 0.0
    if len(bs) > 1:
        var_obs = sum((b - bbar) ** 2 for b in bs) / (len(bs) - 1)
    else:
        var_obs = 0.25
    var_noise = sum(s * s for s in ses) / len(ses) if ses else 0.25
    tau_b = math.sqrt(min(4.0, max(0.04, var_obs - var_noise)))
    out = []
    for b, s in zip(bs, ses):
        pf = 1.0 / (s * s)
        pp = 1.0 / (tau_b * tau_b)
        out.append((b * pf + bbar * pp) / (pf + pp))
    return out


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build_states(case):
    """Collapse the read history to distinct page states.

    Each state is (t_hours, players, rates, weight): the first read that
    showed the page, and a weight from how long the page stood, capped,
    with a floor.
    """
    hist = case["history"]
    at_h = parse_time(case["at"])
    order = []
    seen = {}
    spans = {}
    cur_key = None
    cur_start = None
    for r in hist:
        t = parse_time(r["at"])
        key = (r["players_tracked"], tuple(r["rates"]))
        if key != cur_key:
            if cur_key is not None:
                spans[cur_key] = spans.get(cur_key, 0.0) + max(0.0, t - cur_start)
            cur_key, cur_start = key, t
            if key not in seen:
                seen[key] = (t, r["players_tracked"], list(r["rates"]))
                order.append(key)
    if cur_key is not None:
        spans[cur_key] = spans.get(cur_key, 0.0) + max(0.0, at_h - cur_start)
    states = []
    for key in order:
        t, players, rates = seen[key]
        w = min(spans.get(key, 0.5), 4.0) + 0.25
        states.append((t, players, rates, w))
    if len(states) > MAX_STATES:
        step = len(states) / MAX_STATES
        thin = [states[min(len(states) - 1, int(i * step))] for i in range(MAX_STATES)]
        thin[-1] = states[-1]
        states = thin
    return states


def anchored_predict(fit, b_shrunk, x_last, x_target, level_last, damp):
    """Level-anchored, damped, capped extrapolation of a fitted line."""
    a = fit.a + (fit.b - b_shrunk) * fit.xbar
    fit_last = a + b_shrunk * x_last
    level = fit_last + ANCHOR * (level_last - fit_last)
    inc = damp * b_shrunk * (x_target - x_last)
    inc = min(2.5, max(-2.5, inc))
    return level + inc


def recent_level(pairs):
    """Unweighted mean of the last up-to-3 (value, weight) pairs: a page
    that stood long is one observation, not several, and must not pull the
    anchor back for a fast-moving trophy."""
    recent = pairs[-3:]
    return sum(v for v, _ in recent) / len(recent)


def fit_predict(states, target_h, trophies_n):
    """Fit the model on states; predict every trophy's median logit at
    target_h. Returns (logits, extras)."""
    t_last = states[-1][0]
    x_last, x_target = x_of(t_last), x_of(target_h)
    win_idx = []
    for h in WINDOWS:
        idx = [j for j, s in enumerate(states) if s[0] >= t_last - h]
        if len(idx) < 3:
            idx = list(range(max(0, len(states) - 8), len(states)))
        win_idx.append(idx)
    wins = [[states[j] for j in idx] for idx in win_idx]

    # Players curve per window (no cross-trophy shrinkage: one series),
    # anchored to the same recent level the trophies use.
    ln_p_level = recent_level([(math.log(max(1, s[1])), s[3]) for s in states])
    ln_p_preds = []
    for sub in wins:
        xs = [x_of(s[0]) for s in sub]
        fit = wls(xs, [math.log(max(1, s[1])) for s in sub], [s[3] for s in sub])
        ln_p_preds.append(anchored_predict(fit, fit.b, x_last, x_target,
                                           ln_p_level, DAMP_PLAYERS))
    ln_p_pred = median(ln_p_preds)

    # Per-trophy series.
    e_series, r_series = [], []
    for i in range(trophies_n):
        es, rs = [], []
        for (t, p, rates, w) in states:
            r = rates[i] if i < len(rates) else None
            es.append(None if r is None else max(r, 0.0) * p / 100.0)
            rs.append(r)
        e_series.append(es)
        r_series.append(rs)

    # Fits per view and window; slopes shrunk across trophies per group.
    # pred[view][i] = list of per-window predicted log quantities.
    pred = {"e": [[] for _ in range(trophies_n)], "r": [[] for _ in range(trophies_n)]}
    resid = {"e": [[] for _ in range(trophies_n)], "r": [[] for _ in range(trophies_n)]}
    seb_e = [[] for _ in range(trophies_n)]
    seb_r = [[] for _ in range(trophies_n)]
    seb = {"e": seb_e, "r": seb_r}
    for view, series, damp in (("e", e_series, DAMP_EARNERS), ("r", r_series, 1.0)):
        for wi, sub in enumerate(wins):
            idx = win_idx[wi]
            fits, owners = [], []
            for i in range(trophies_n):
                xs, ys, ws = [], [], []
                for j in idx:
                    v = series[i][j]
                    if v is None:
                        continue
                    xs.append(x_of(states[j][0]))
                    ys.append(math.log(max(v, 0.3)) if view == "e" else logit(v))
                    ws.append(states[j][3])
                if len(xs) >= 2:
                    fits.append(wls(xs, ys, ws))
                    owners.append(i)
            slopes = shrink_slopes(fits) if fits else []
            for i, fit, b in zip(owners, fits, slopes):
                pairs = [(math.log(max(series[i][j], 0.3)) if view == "e" else logit(series[i][j]),
                          states[j][3]) for j in idx if series[i][j] is not None]
                lvl = recent_level(pairs)
                pred[view][i].append(anchored_predict(fit, b, x_last, x_target, lvl, damp))
                resid[view][i].append(fit.resid)
                seb[view][i].append(fit.se_b)

    logits, out_resid, out_seb, out_dis, e_last, present_last = [], [], [], [], [], []
    for i in range(trophies_n):
        views = []
        all_preds = []
        if pred["e"][i]:
            per_w = [logit(100.0 * math.exp(v - ln_p_pred)) for v in pred["e"][i]]
            all_preds += per_w
            views.append((median(per_w), W_EARNERS_VIEW))
        if pred["r"][i]:
            all_preds += pred["r"][i]
            views.append((median(pred["r"][i]), 1.0 - W_EARNERS_VIEW))
        if views:
            wsum = sum(w for _, w in views)
            logits.append(sum(v * w for v, w in views) / wsum)
        else:
            logits.append(None)
        if len(all_preds) >= 2:
            m = sum(all_preds) / len(all_preds)
            out_dis.append(sum((v - m) ** 2 for v in all_preds) / len(all_preds))
        else:
            out_dis.append(0.0)
        res = resid["e"][i] + resid["r"][i]
        out_resid.append(median(res) if res else 1.0)
        seb = seb_e[i] + seb_r[i]
        out_seb.append(median(seb) if seb else 0.5)
        pres = [e for e in e_series[i] if e is not None]
        e_last.append(pres[-1] if pres else None)
        present_last.append(bool(e_series[i] and e_series[i][-1] is not None))
    extras = {"resid": out_resid, "se_b": out_seb, "disagree": out_dis,
              "e_last": e_last, "present_last": present_last}
    return logits, extras


def stale_page_forecast(case, states):
    """One frozen page in the whole history: the page is stale, and the
    contest's own copy documents the refresh that follows -- the tracked
    crowd multiplies (638 to 3,868 at the first refresh) while earners,
    which need play time, lag. Dilute by that multiplier; a trophy at
    effectively 100% is held by every tracked player and stays so."""
    last = states[-1]
    players = max(1, last[1])
    rates = last[2]
    mult = 5.4
    sigma = 0.7
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = rates[i] if i < len(rates) else None
        if r is None:
            med = 0.05
        elif r >= 99.99:
            med = 100.0
        else:
            med = 100.0 * (r * players / 100.0) / (players * mult)
        rows.append((med, sigma))
    return rows


def persistence_rows(case):
    """The starter's forecast: last rate, random-walk spread."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i] if i < len(last["rates"]) else None
        rows.append((r if r is not None else 0.0, spread))
    return rows


def rows_to_output(trophies, med_sigma, zs):
    rows = []
    for trophy, (med, sigma) in zip(trophies, med_sigma):
        lg = logit(med)
        qs = []
        prev = 0.0
        for z in zs:
            q = min(100.0, max(0.0, inv_logit(lg + z * sigma)))
            q = max(q, prev)
            prev = q
            qs.append(round4(q))
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    zs = [z_for(lv) for lv in levels]
    trophies = case["trophies"]
    if not case.get("history"):
        return rows_to_output(trophies, [(0.0, 0.5) for _ in trophies], zs)
    states = build_states(case)
    target_h = parse_time(case["target"])
    span = states[-1][0] - states[0][0] if len(states) > 1 else 0.0
    if len(states) < 3 or span < 1.0:
        return rows_to_output(trophies, stale_page_forecast(case, states), zs)

    logits, extras = fit_predict(states, target_h, len(trophies))
    dx = x_of(target_h) - x_of(states[-1][0])
    rows = []
    for i, trophy in enumerate(trophies):
        # Last observed rate of this trophy (scanning back over nulls).
        last_obs = None
        for s in reversed(states):
            if i < len(s[2]) and s[2][i] is not None:
                last_obs = s[2][i]
                break
        lg = logits[i]
        if lg is None:
            lg = logit(last_obs) if last_obs is not None else logit(0.05)
            sigma = 1.2
        else:
            # No line runs to 100% or to 0 in a week: bound the median's
            # move from the last observed rate.
            if last_obs is not None:
                lo = logit(last_obs) - 3.0
                hi = logit(last_obs) + 2.2
                lg = min(hi, max(lo, lg))
            var = (extras["resid"][i] ** 2
                   + (max(extras["se_b"][i], 0.08) * dx) ** 2
                   + 0.3 ** 2
                   + 0.5 * extras["disagree"][i])
            sigma = SIGMA_CAL * math.sqrt(var)
            if last_obs is not None and last_obs <= 0.0:
                sigma = max(sigma, 1.1)
            e = extras["e_last"][i]
            if e is not None and e < 3.0:
                sigma *= 1.35
            elif e is not None and e < 10.0:
                sigma *= 1.15
            if not extras["present_last"][i]:
                sigma *= 1.3
            sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        rows.append((inv_logit(lg), sigma))
    return rows_to_output(trophies, rows, zs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = rows_to_output(
                case["trophies"], persistence_rows(case),
                [z_for(lv) for lv in (case.get("levels") or LEVELS)])
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
