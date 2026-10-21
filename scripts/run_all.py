"""All experiments; writes results/summary.json, results/daily.csv and figures/.
Usage: python scripts/run_all.py [n_workers]"""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eafsched import heuristics as H  # noqa: E402
from eafsched import model as M  # noqa: E402
from eafsched import prices as Pr  # noqa: E402
from eafsched import sim as S  # noqa: E402
from eafsched.dp import solve_dp  # noqa: E402
from eafsched.plant import Plant  # noqa: E402

BASE = Plant()
I0 = 144                    # 1.5 heats in the buffer at the start of every day (and at least that at its end)
QS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DAYS = None


def day_base(i):
    d, p = DAYS[i]
    row = dict(date=str(d.date()), n=len(p), mean_price=float(p.mean()))
    st = H.steady(p, BASE, I0)
    c = M.check_schedule(st, BASE, I0, p)
    row.update(steady=c["cost"], steady_ok=c["feasible"], energy=c["energy_mwh"])
    for q in QS:
        th = H.threshold(p, BASE, I0, q)
        c = M.check_schedule(th, BASE, I0, p)
        row[f"thr{q}"] = c["cost"] if c["feasible"] else np.nan
    v, sdp = solve_dp(p, BASE, I0)
    row["dp"] = v
    row["dp_ok"] = M.check_schedule(sdp, BASE, I0, p)["feasible"]
    m = M.solve_milp(p, BASE, I0)
    row.update(milp=m["cost"], milp_time=m["wall"], milp_ok=M.check_schedule(m["starts"], BASE, I0, p)["feasible"],
               milp_heats=m["heats"])
    if i % 20 == 0:
        lp = M.solve_milp(p, BASE, I0, relax=True)
        row["lp_bound"] = lp["cost"]
    return row


def day_flex(args):
    i, buf_heats, hpd = args
    d, p = DAYS[i]
    pl = replace(BASE, buf_max=int(96 * buf_heats), draw=hpd)
    st = H.steady(p, pl, I0)
    cs = M.check_schedule(st, pl, I0, p)
    v, _ = solve_dp(p, pl, I0)
    return dict(i=i, buf=buf_heats, hpd=hpd, steady=cs["cost"] if cs["feasible"] else np.nan, dp=v)


def day_sim(args):
    i, seed = args
    d, p = DAYS[i]
    out = dict(i=i, seed=seed)
    for pol, saf in [("steady", 0), ("static", 0), ("static", 24), ("static", 48), ("static", 72), ("reopt", 0),
                     ("reopt", 48)]:
        rng = np.random.default_rng(seed * 100003 + i)
        r = S.run_day(p, BASE, I0, pol, rng, p_delay=0.15, safety=saf)
        k = f"{pol}_s{saf}"
        out[k + "_cost"], out[k + "_stops"], out[k + "_short"] = r["cost"], r["stops"], r["end_short"]
        # steel not melted by the end of the day still has to be melted: charge it at the day's mean price
        out[k + "_cost"] += r["end_short"] / BASE.unit * BASE.energy_per_heat() * p.mean()
    return out


def two_furnaces(i, cap):
    d, p = DAYS[i]
    pl = replace(BASE, n_furnaces=2, draw=28, buf_max=288, power_cap=cap)
    m = M.solve_milp(p, pl, 192, time_limit=120)        # two heats opening stock: 28 heats/day drain 112 units per cycle
    if "starts" not in m:
        return dict(i=i, cap=cap, cost=np.nan, time=m["wall"], nodes=0, gap=np.nan, ok=False, peak_mw=np.nan)
    chk = M.check_schedule(m["starts"], pl, 192, p)
    return dict(i=i, cap=cap, cost=m["cost"], time=m["wall"], nodes=m["nodes"], gap=m["gap"], ok=chk["feasible"],
                peak_mw=float(m["mw"].max()))


def init(days):
    global DAYS
    DAYS = days


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    s = Pr.load()
    # the six daylight-saving days (23 h or 25 h) are excluded: the caster then draws 19.2 or 20.8 heats, so a day
    # cannot close at its opening stock without breaking the buffer limits in some configurations
    days = [(d, p) for d, p in Pr.days(s) if len(p) == 96]
    init(days)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    out = dict(n_days=len(days), first=str(days[0][0].date()), last=str(days[-1][0].date()))
    t0 = time.time()
    with Pool(nw, initializer=init, initargs=(days,)) as pool:
        base = pd.DataFrame(pool.map(day_base, range(len(days)), chunksize=8))
        print("base done", round(time.time() - t0), flush=True)
        base["year"] = base.date.str[:4].astype(int)
        base.to_csv(ROOT / "results" / "daily.csv", index=False)
        # threshold rule tuned on 2023, tested on 2024-2025
        q_best = min(QS, key=lambda q: base.loc[base.year == 2023, f"thr{q}"].sum())
        out["threshold_q_tuned_on_2023"] = q_best
        yr = {}
        for y, g in base.groupby("year"):
            stc = g.steady.sum()
            yr[int(y)] = dict(days=len(g), energy_GWh=g.energy.sum() / 1e3, steady_MEUR=stc / 1e6,
                              threshold_MEUR=g[f"thr{q_best}"].sum() / 1e6, optimal_MEUR=g.milp.sum() / 1e6,
                              saving_optimal_pct=100 * (1 - g.milp.sum() / stc),
                              saving_threshold_pct=100 * (1 - g[f"thr{q_best}"].sum() / stc),
                              saving_optimal_MEUR=(stc - g.milp.sum()) / 1e6,
                              price_paid_steady=stc / g.energy.sum(), price_paid_optimal=g.milp.sum() / g.energy.sum(),
                              mean_price=float(g.mean_price.mean()))
        out["by_year"] = yr
        out["verification"] = dict(
            max_abs_milp_minus_dp=float((base.milp - base.dp).abs().max()),
            all_milp_feasible=bool(base.milp_ok.all()), all_dp_feasible=bool(base.dp_ok.all()),
            all_steady_feasible=bool(base.steady_ok.all()),
            threshold_infeasible_days={str(q): int(base[f"thr{q}"].isna().sum()) for q in QS},
            milp_time_median_s=float(base.milp_time.median()), milp_time_max_s=float(base.milp_time.max()),
            lp_relaxation_gap_pct_median=float(np.nanmedian(100 * (base.milp - base.lp_bound) / base.milp)),
            lp_relaxation_gap_pct_max=float(np.nanmax(100 * (base.milp - base.lp_bound) / base.milp)))
        print(json.dumps(out, indent=1), flush=True)

        # flexibility: buffer size x caster rate (2024-2025)
        idx = [i for i, (d, _) in enumerate(days) if d.year >= 2024]
        jobs = [(i, b, h) for b in (1.5, 2, 3, 4) for h in (16, 18, 20, 22) for i in idx]
        flex = pd.DataFrame(pool.map(day_flex, jobs, chunksize=32))
        f = flex.groupby(["buf", "hpd"])[["steady", "dp"]].sum()
        f["saving_pct"] = 100 * (1 - f.dp / f.steady)
        f["saving_MEUR_per_yr"] = (f.steady - f.dp) / 1e6 / (len(idx) / 365.25)
        out["flexibility"] = {f"{b}heats_{h}perday": dict(saving_pct=float(r.saving_pct),
                                                          saving_MEUR_per_yr=float(r.saving_MEUR_per_yr))
                              for (b, h), r in f.iterrows()}
        print("flex done", round(time.time() - t0), flush=True)

        # delays: 2024, 10 seeds
        idx24 = [i for i, (d, _) in enumerate(days) if d.year == 2024]
        simr = pd.DataFrame(pool.map(day_sim, [(i, s_) for s_ in range(10) for i in idx24], chunksize=16))
        pol = {}
        for k in ["steady_s0", "static_s0", "static_s24", "static_s48", "static_s72", "reopt_s0", "reopt_s48"]:
            per = simr.groupby("seed")[[k + "_cost", k + "_stops", k + "_short"]].sum()
            pol[k] = dict(cost_MEUR=float(per[k + "_cost"].mean() / 1e6), stop_slots_per_yr=float(per[k + "_stops"].mean()),
                          days_with_stop=float((simr[k + "_stops"] > 0).sum() / 10),
                          end_shortfall_heats_per_yr=float(per[k + "_short"].mean() / 96))
        out["delays_2024_p015"] = pol
        print("sim done", round(time.time() - t0), flush=True)

    # two furnaces with a grid limit (MILP only), 30 days of 2024
    sample = idx24[::12][:30]
    tf = pd.DataFrame([two_furnaces(i, cap) for i in sample for cap in (1e9, 80.0)])
    g = tf.groupby("cap")
    out["two_furnaces"] = {("no_cap" if c > 1e8 else f"cap_{int(c)}MW"): dict(
        cost_MEUR_30d=float(r.cost.sum() / 1e6), time_median_s=float(r.time.median()), time_max_s=float(r.time.max()),
        nodes_median=float(r.nodes.median()), all_feasible=bool(r.ok.all()), peak_mw_max=float(r.peak_mw.max()))
        for c, r in g}
    out["two_furnaces"]["cap_cost_pct"] = float(100 * (tf[tf.cap < 1e8].cost.sum() / tf[tf.cap > 1e8].cost.sum() - 1))
    (ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    figures(base, f, days)


def figures(base, f, days):
    # one example day
    i = len(days) // 2 + 40
    d, p = days[i]
    st = H.steady(p, BASE, I0)
    v, opt = solve_dp(p, BASE, I0)
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    t = np.arange(len(p)) / 4
    ax[0].step(t, p, where="post", color="k"); ax[0].set_ylabel("EUR/MWh")
    ax[0].set_title(f"IT-North day-ahead, {d.date()}")
    for lab, sch, col in [("price-blind", st, "0.6"), ("optimal", opt, "tab:red")]:
        c = M.check_schedule(sch, BASE, I0, p)
        mw = np.zeros(len(p))
        for s in np.nonzero(sch)[0]:
            mw[s:s + BASE.cycle] += BASE.profile()
        ax[1].step(t, mw, where="post", color=col, label=f"{lab}: {c['cost'] / 1e3:.1f} kEUR")
        inv = I0 - BASE.draw * np.arange(len(p) + 1)
        for s in np.nonzero(sch)[0]:
            inv[s + BASE.cycle:] += BASE.unit
        ax[2].step(np.arange(len(p) + 1) / 4, inv / 96, where="post", color=col)
    ax[1].set_ylabel("furnace MW"); ax[1].legend(fontsize=8, frameon=True)
    ax[2].set_ylabel("buffer (heats)"); ax[2].set_xlabel("hour")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "example_day.png", dpi=150); plt.close(fig)
    # flexibility heat map
    piv = f.saving_pct.unstack("hpd")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = ax.imshow(piv.values, cmap="jet", origin="lower", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("caster demand (heats/day)"); ax.set_ylabel("buffer (heats)")
    for a in range(piv.shape[0]):
        for b in range(piv.shape[1]):
            ax.text(b, a, f"{piv.values[a, b]:.1f}", ha="center", va="center", color="w", fontsize=9)
    plt.colorbar(im, label="saving vs price-blind (%)")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "flexibility.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
