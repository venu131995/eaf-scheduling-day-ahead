"""Day-ahead heat scheduling as a mixed-integer linear programme (HiGHS through scipy.optimize.milp).

x[f, t] = 1 if furnace f starts a heat in slot t. A heat occupies slots t .. t+cycle-1 and its steel reaches the
buffer at the end of slot t+cycle-1. With A_t = heats finished by the end of slot t (all furnaces), the buffer after
slot t is I_t = I0 + unit A_t - draw t. Within slot t the caster draws before the new steel arrives, so the
low point is I0 + unit A_{t-1} - draw t >= buf_min; after arrival I_t <= buf_max. The day must end with at least
its opening stock (no end-of-horizon gaming).
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_milp(price, plant, I0, time_limit=60.0, relax=False):
    """price: EUR/MWh per slot (length n). Returns dict(cost, starts[f, t], mw[t], inventory[t], status, ...)."""
    price = np.asarray(price, float)
    n, F, c = len(price), plant.n_furnaces, plant.cycle
    prof = np.array(plant.profile())
    nv = F * n
    idx = lambda f, t: f * n + t  # noqa: E731

    # objective: cost of each start = sum_k price[t+k] * prof[k] * 0.25
    cost = np.zeros(nv)
    for f in range(F):
        for t in range(n - c + 1):
            cost[idx(f, t)] = 0.25 * np.dot(price[t:t + c], prof)
    ub = np.zeros(nv)
    for f in range(F):
        ub[idx(f, 0):idx(f, n - c) + 1] = 1.0

    rows, lo, hi = [], [], []
    A = lil_matrix((F * n + 3 * n + 1 + max(F - 1, 0), nv))
    r = 0
    for f in range(F):                     # no overlap on a furnace
        for t in range(n):
            for k in range(c):
                if t - k >= 0:
                    A[r, idx(f, t - k)] = 1
            lo.append(-np.inf); hi.append(1); r += 1
    for t in range(1, n + 1):              # buffer bounds
        # finished by end of slot t-1 (low point) and by end of slot t (after arrival)
        for f in range(F):
            for tau in range(0, t - c + 1):          # tau + c <= t  -> finished by end of slot t
                A[r + 1, idx(f, tau)] = plant.unit
                if tau + c <= t - 1:
                    A[r, idx(f, tau)] = plant.unit
        lo.append(plant.buf_min - I0 + plant.draw * t); hi.append(np.inf)          # low point
        lo.append(-np.inf); hi.append(plant.buf_max - I0 + plant.draw * t)         # after arrival
        r += 2
    for t in range(n):                     # grid power cap
        for f in range(F):
            for k in range(c):
                if t - k >= 0:
                    A[r, idx(f, t - k)] = prof[k]
        lo.append(-np.inf); hi.append(plant.power_cap); r += 1
    for f in range(F):                     # end of day: stock at least the opening stock
        for tau in range(n - c + 1):
            A[r, idx(f, tau)] = plant.unit
    lo.append(plant.draw * n); hi.append(np.inf); r += 1
    for f in range(F - 1):                 # symmetry breaking between identical furnaces
        for t in range(n):
            A[r, idx(f, t)] = 1
            A[r, idx(f + 1, t)] = -1
        lo.append(0); hi.append(np.inf); r += 1
    A = A[:r].tocsr()
    t0 = time.perf_counter()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(np.zeros(nv), ub),
               integrality=None if relax else np.ones(nv),
               options=dict(time_limit=time_limit, mip_rel_gap=1e-9, presolve=True))
    wall = time.perf_counter() - t0
    if res.x is None:
        return dict(status=res.status, message=res.message, wall=wall)
    x = res.x.reshape(F, n)
    if not relax:
        x = np.round(x)
    mw = np.zeros(n)
    for f in range(F):
        for t in range(n):
            for k in range(c):
                if t + k < n:
                    mw[t + k] += prof[k] * x[f, t]
    finished = np.zeros(n + 1)
    for f in range(F):
        for t in range(n):
            if x[f, t] > 0 and t + c <= n:
                finished[t + c:] += x[f, t]
    inv = I0 + plant.unit * finished - plant.draw * np.arange(n + 1)
    return dict(status=res.status, cost=float(res.fun), starts=x, mw=mw, inventory=inv, wall=wall,
                gap=float(getattr(res, "mip_gap", 0.0) or 0.0), nodes=int(getattr(res, "mip_node_count", 0) or 0),
                heats=float(x.sum()))


def check_schedule(starts, plant, I0, price):
    """Independent feasibility and cost check of a schedule (1 or more furnaces)."""
    starts = np.atleast_2d(starts)
    F, n = starts.shape
    c = plant.cycle
    prof = np.array(plant.profile())
    viol = []
    mw = np.zeros(n)
    busy = np.zeros((F, n))
    arrivals = np.zeros(n)
    for f in range(F):
        for t in np.nonzero(starts[f] > 0.5)[0]:
            if t + c > n:
                viol.append(f"heat on furnace {f} at slot {t} ends after the day")
                continue
            busy[f, t:t + c] += 1
            mw[t:t + c] += prof
            arrivals[t + c - 1] += 1
    if (busy > 1).any():
        viol.append("overlapping heats on a furnace")
    if (mw > plant.power_cap + 1e-9).any():
        viol.append("power cap exceeded")
    I = I0
    for t in range(n):
        I -= plant.draw
        if I < plant.buf_min - 1e-9:
            viol.append(f"buffer below minimum in slot {t}")
            break
        I += plant.unit * arrivals[t]
        if I > plant.buf_max + 1e-9:
            viol.append(f"buffer above maximum in slot {t}")
            break
    if I < I0 - 1e-9:
        viol.append("day ends below opening stock")
    return dict(feasible=not viol, violations=viol, cost=float(0.25 * np.dot(price, mw)), end_inventory=I,
                energy_mwh=float(0.25 * mw.sum()))
