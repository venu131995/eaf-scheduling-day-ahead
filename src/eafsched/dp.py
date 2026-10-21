"""Exact dynamic programme for one furnace, used to certify the MILP optimum and as a fast solver.

State at the start of slot t: buffer level I (units) and b = slots of the current heat still to run (0 = idle).
In slot t the caster draws first (I - draw >= buf_min), the furnace runs, and if its heat ends in this slot the
steel arrives (I + unit <= buf_max). The day must end idle with I_n >= target (default: the opening stock).
The recursion is vectorised over the buffer level.
"""
import numpy as np


def _step(V1, price_t, plant, prof, b, start, lo):
    """Value over all buffer levels I of taking `start` in state b at a slot with price price_t."""
    hi, c = plant.buf_max, plant.cycle
    I = np.arange(hi + 1)
    bb = c if start else b
    I1 = I - plant.draw
    cost = 0.0
    arrive = False
    if bb > 0:
        cost = 0.25 * price_t * prof[c - bb]
        bb -= 1
        arrive = bb == 0
    if arrive:
        I1 = I1 + plant.unit
    ok = (I - plant.draw >= lo) & (I1 <= hi)
    out = np.full(hi + 1, np.inf)
    out[ok] = cost + V1[I1[ok], bb]
    return out


def solve_dp(price, plant, I0, b0=0, target=None):
    """Optimal cost and starts. b0: slots still to run of a heat already in progress at slot 0 (its profile
    continues); target: minimum closing stock (default I0). The safety stock buf_min applies from the end of the
    first cycle on, since no heat can arrive earlier. Returns (inf, zeros) if the day is infeasible."""
    assert plant.n_furnaces == 1
    target = I0 if target is None else target
    price = np.asarray(price, float)
    n, c = len(price), plant.cycle
    prof = np.array(plant.profile())
    hi = plant.buf_max
    V = np.full((n + 1, hi + 1, c + 1), np.inf)
    V[n, max(target, plant.buf_min):, 0] = 0.0
    choice = np.zeros((n, hi + 1), np.int8)          # start decision in the idle state
    for t in range(n - 1, -1, -1):
        lo = plant.buf_min if t >= c else 0
        for b in range(1, c + 1):
            V[t, :, b] = _step(V[t + 1], price[t], plant, prof, b, False, lo)
        idle = _step(V[t + 1], price[t], plant, prof, 0, False, lo)
        go = _step(V[t + 1], price[t], plant, prof, 0, True, lo)
        choice[t] = go < idle
        V[t, :, 0] = np.minimum(idle, go)
    starts = np.zeros(n)
    if not np.isfinite(V[0, I0, b0]):
        return np.inf, starts
    I, b = I0, b0
    for t in range(n):
        if b == 0 and choice[t, I]:
            starts[t] = 1
            b = c
        I -= plant.draw
        if b > 0:
            b -= 1
            if b == 0:
                I += plant.unit
    return float(V[0, I0, b0]), starts
