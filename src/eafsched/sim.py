"""Executing a day's schedule when melting runs late.

Each heat's melting takes one extra quarter-hour with probability p (independently), drawing melting power for
that slot too. Policies:
  steady: the price-blind schedule (heuristics.steady), executed like a static plan.
  static: follow the day-ahead plan; a heat whose slot is taken by a late predecessor starts as soon as the
          furnace is free.
  reopt:  whenever the furnace becomes free, re-solve the dynamic programme for the rest of the day from the
          current buffer level and start a heat if the new plan says so.
Plans can hold a safety stock (buf_min > 0); execution itself only stops the caster when the buffer is empty.
A caster stop is a slot in which the buffer cannot supply the caster's draw.
"""
from dataclasses import replace

import numpy as np

from .dp import solve_dp
from .heuristics import steady


def run_day(price, plant, I0, policy, rng, p_delay=0.15, safety=0):
    n, c = len(price), plant.cycle
    plan_plant = replace(plant, buf_min=safety)
    plan = None
    if policy == "static":
        v, plan = solve_dp(price, plan_plant, I0)
        if not np.isfinite(v):
            plan = solve_dp(price, plant, I0)[1]
    elif policy == "steady":
        plan = steady(price, plant, I0)
    I, cost, stops = I0, 0.0, 0
    t_free = 0                       # first slot at which the furnace is free
    heat = None                      # (end slot exclusive, melting slots) of the heat in progress
    queue = list(np.nonzero(plan)[0]) if plan is not None else []
    t = 0
    while t < n:
        if heat is None and t >= t_free and t + c <= n:
            start = False
            if policy in ("static", "steady"):
                if queue and queue[0] <= t:
                    queue.pop(0)
                    start = True
            else:
                need = plant.draw * (n - t) - (I - I0) > 0
                if need:
                    v, st = solve_dp(price[t:], plan_plant, I, 0, target=I0)
                    if not np.isfinite(v):  # target unreachable: fall back to a plan without the safety stock
                        v, st = solve_dp(price[t:], replace(plant, buf_min=0), max(I, 0), 0, target=0)
                    # if even that is infeasible a stop is already unavoidable: melt now to shorten it
                    start = bool(st[0] > 0) if np.isfinite(v) else True
            if start:
                melt = plant.power_on + int(rng.random() < p_delay)
                heat = (t + melt + plant.tap, melt)
                h0 = t
        # power this slot
        if heat is not None:
            k = t - h0
            cost += 0.25 * price[t] * (plant.mw_melt if k < heat[1] else plant.mw_tap)
        # caster draw
        if I >= plant.draw:
            I -= plant.draw
        else:
            stops += 1
        # heat finishes at the end of this slot
        if heat is not None and t + 1 == heat[0]:
            I += plant.unit
            t_free = t + 1
            heat = None
        t += 1
    return dict(cost=cost, stops=stops, end_short=max(I0 - I, 0), end_inventory=I)
