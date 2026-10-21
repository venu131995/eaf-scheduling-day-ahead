"""Price-blind and rule-based schedules for one furnace, the benchmarks for the optimiser.

* steady: start a heat as soon as the furnace is idle and the steel would fit in the buffer when it arrives
  (the plant runs ahead of the caster and then waits): the price-blind way a melt shop runs.
* threshold: the same, but while there is stock to spare a heat is started only if its energy-weighted price is in
  the cheapest q-quantile of the day's remaining start windows. A heat is forced when waiting one more slot would
  let the buffer run dry before the next heat could arrive, or leave too little time to close the day at the
  opening stock.
"""
import numpy as np


def _run(price, plant, I0, want):
    n, c = len(price), plant.cycle
    prof = np.array(plant.profile())
    starts = np.zeros(n)
    I, b = I0, 0
    for t in range(n):
        if b == 0 and t + c <= n:
            # stock at the arrival of a heat started now (after its arrival) and its low point before arrival
            fits = I - plant.draw * c + plant.unit <= plant.buf_max
            # if we wait one slot, the next heat arrives at the end of slot t+c: low point I - draw (c+1)
            forced = I - plant.draw * (c + 1) < plant.buf_min
            short = plant.draw * (n - t) - (I - I0)          # units still to melt to close at the opening stock
            need = -(-short // plant.unit) if short > 0 else 0  # heats still needed
            # waiting one more slot would leave too little time to melt them all before the day ends
            forced = forced or (need > 0 and n - t - 1 < need * c)
            if fits and need > 0 and (forced or want(t, I)):
                starts[t] = 1
                b = c
        I -= plant.draw
        if b > 0:
            b -= 1
            if b == 0:
                I += plant.unit
    return starts


def steady(price, plant, I0):
    return _run(price, plant, I0, lambda t, I: True)


def threshold(price, plant, I0, q=0.4):
    price = np.asarray(price, float)
    n, c = len(price), plant.cycle
    prof = np.array(plant.profile())
    w = np.array([np.dot(price[t:t + c], prof) if t + c <= n else np.inf for t in range(n)])

    def want(t, I):
        rest = w[t:n - c + 1]
        return w[t] <= np.quantile(rest, q) if len(rest) else True

    return _run(price, plant, I0, want)
