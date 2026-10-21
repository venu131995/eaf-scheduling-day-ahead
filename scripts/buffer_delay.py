"""Delays (p = 0.15) with a larger buffer: is the saving robust once the plant has room for a safety stock?
Writes results/buffer_delay.json. Static day-ahead plan with safety stock s, price-blind schedule as reference."""
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eafsched import prices as Pr  # noqa: E402
from eafsched import sim as S  # noqa: E402
from eafsched.plant import Plant  # noqa: E402

DAYS = [(d, p) for d, p in Pr.days(Pr.load()) if len(p) == 96 and d.year == 2024]
CASES = [(buf, saf) for buf in (2, 3) for saf in (0, 48, 72, 96)]


def one(args):
    i, seed = args
    p = DAYS[i][1]
    out = {}
    for buf, saf in CASES:
        pl = replace(Plant(), buf_max=96 * buf)
        I0 = 144 if buf == 2 else 192
        for pol in ("steady", "static"):
            if pol == "steady" and saf:
                continue
            r = S.run_day(p, pl, I0, pol, np.random.default_rng(seed * 100003 + i), 0.15, saf)
            cost = r["cost"] + r["end_short"] / pl.unit * pl.energy_per_heat() * p.mean()
            out[(buf, saf, pol)] = (cost, r["stops"], r["end_short"] / 96)
    return out


if __name__ == "__main__":
    jobs = [(i, s) for s in range(5) for i in range(len(DAYS))]
    with Pool(6) as pool:
        res = pool.map(one, jobs, chunksize=16)
    agg = {}
    for r in res:
        for k, v in r.items():
            a = agg.setdefault(k, np.zeros(3))
            a += np.array(v)
    out = {}
    for (buf, saf, pol), v in agg.items():
        out[f"buf{buf}_s{saf}_{pol}"] = dict(cost_MEUR=v[0] / 5 / 1e6, stop_slots_per_yr=v[1] / 5, shortfall_heats_per_yr=v[2] / 5)
    for buf in (2, 3):
        ref = out[f"buf{buf}_s0_steady"]["cost_MEUR"]
        for saf in (0, 48, 72, 96):
            k = f"buf{buf}_s{saf}_static"
            out[k]["saving_vs_price_blind_pct"] = 100 * (1 - out[k]["cost_MEUR"] / ref)
    (ROOT / "results" / "buffer_delay.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
