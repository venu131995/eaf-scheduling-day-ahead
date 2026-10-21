import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eafsched import heuristics as H  # noqa: E402
from eafsched import model as M  # noqa: E402
from eafsched import sim as S  # noqa: E402
from eafsched.dp import solve_dp  # noqa: E402
from eafsched.plant import Plant  # noqa: E402

BASE = Plant()
I0 = 144


def prices(seed, n=96):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 4
    return 90 + 30 * np.sin(2 * np.pi * (t - 8) / 24) + 15 * rng.standard_normal(n)


def test_dp_equals_brute_force_on_small_days():
    # 24-slot day, caster 4 units per slot of a 24-unit heat -> 4 heats needed (slack 8 slots)
    pl = replace(BASE, unit=24, draw=4, buf_max=40, buf_min=0)
    for seed in range(5):
        p = prices(seed, 24)
        best = np.inf
        n, c = 24, pl.cycle
        for k in range(3, 7):
            for comb in itertools.combinations(range(n - c + 1), k):
                st = np.zeros(n)
                st[list(comb)] = 1
                chk = M.check_schedule(st, pl, 20, p)
                if chk["feasible"]:
                    best = min(best, chk["cost"])
        v, st = solve_dp(p, pl, 20)
        assert abs(v - best) < 1e-9
        assert M.check_schedule(st, pl, 20, p)["feasible"]


def test_milp_equals_dp_and_is_feasible():
    for seed in range(4):
        p = prices(seed)
        m = M.solve_milp(p, BASE, I0)
        v, st = solve_dp(p, BASE, I0)
        assert abs(m["cost"] - v) < 1e-6
        assert M.check_schedule(m["starts"], BASE, I0, p)["feasible"]
        assert abs(M.check_schedule(st, BASE, I0, p)["cost"] - v) < 1e-6


def test_lp_relaxation_is_a_lower_bound():
    p = prices(7)
    assert M.solve_milp(p, BASE, I0, relax=True)["cost"] <= M.solve_milp(p, BASE, I0)["cost"] + 1e-6


def test_heuristics_feasible_and_not_better_than_optimum():
    for seed in range(4):
        p = prices(seed)
        v, _ = solve_dp(p, BASE, I0)
        for sch in (H.steady(p, BASE, I0), H.threshold(p, BASE, I0, 0.4)):
            c = M.check_schedule(sch, BASE, I0, p)
            assert c["feasible"] and c["cost"] >= v - 1e-6


def test_constant_price_cost_is_energy_times_price():
    p = np.full(96, 100.0)
    v, st = solve_dp(p, BASE, I0)
    assert abs(v - st.sum() * BASE.energy_per_heat() * 100.0) < 1e-6


def test_two_furnaces_power_cap_respected():
    pl = replace(BASE, n_furnaces=2, draw=28, buf_max=288, power_cap=80.0)
    p = prices(3)
    m = M.solve_milp(p, pl, 192)
    c = M.check_schedule(m["starts"], pl, 192, p)
    assert c["feasible"] and m["mw"].max() <= 80.0 + 1e-9


def test_simulation_without_delays_reproduces_the_plan():
    p = prices(5)
    v, _ = solve_dp(p, BASE, I0)
    rng = np.random.default_rng(0)
    for pol in ("static", "reopt"):
        r = S.run_day(p, BASE, I0, pol, rng, p_delay=0.0)
        assert r["stops"] == 0 and r["end_short"] == 0
        assert abs(r["cost"] - v) < 1e-6
