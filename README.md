# Scheduling an Electric Arc Furnace against Italian Day-Ahead Prices: MILP, Exact DP, Heuristics and What Delays Do

An electric-arc-furnace (EAF) melt shop buys its power on the day-ahead market. The caster downstream needs a steady
flow of liquid steel, so the melt shop can shift its heats only as far as a small ladle buffer allows. How much is
price-aware scheduling worth, how simple can the rule be, and does the saving survive real melt-time variability?

**Data:** IT-North (Italian North) day-ahead prices 2023-2025 from the Energy-Charts API (Fraunhofer ISE, CC BY 4.0),
hourly products held over their quarter-hours until 30 Sep 2025 and 15-minute products after. 1,090 days (the 6
daylight-saving days, 23 h or 25 h long, are left out).

**Plant (illustrative, typical EAF figures):** 100-t heats, 40 MWh each (400 kWh/t) melted in 45 min at 53.3 MW, then
15 min of tapping and charging at 2 MW; caster 20 heats/day (2,000 t/day); ladle buffer 2 heats; the day opens and
must close with 1.5 heats in the buffer. Energy: 295 GWh/yr, about 33 MEUR/yr of power.

## In short

| Question | Answer |
|---|---|
| What is optimal scheduling worth, prices known day-ahead? | **3.0-3.3 % of the power bill: 1.04-1.24 MEUR per year** (2023: 1.24, 2024: 1.06, 2025: 1.04) against the price-blind melt shop. The average price paid drops from 107.9 to 104.3 EUR/MWh in 2024. |
| Is the optimum certified? | The MILP (HiGHS) and an independent exact dynamic programme agree to **1.5e-10 EUR on all 1,090 days**; the DP matches brute-force enumeration on small days; every schedule passes an independent feasibility check. MILP solve time 0.03 s (max 0.2 s); LP relaxation gap 1.3 % median. |
| How good is a simple rule? | A price-threshold rule (start a heat only in the cheapest 70 % of the day's remaining windows unless forced), tuned on 2023: **2.2-2.7 %**, 72-80 % of the optimum (2024-2025 out of sample). |
| What limits the saving? | Flexibility. With a 3-heat buffer the saving at 20 heats/day is 4.6 % (1.5 MEUR/yr); at 22 heats/day (furnace nearly saturated) 2.4 %; with a 1.5-heat buffer 1.2 %. |
| Does it survive delays? | **No, not with today's buffer.** If 15 % of heats melt 15 min longer, the optimal plan (which runs the buffer close to empty) stops the caster on **199 days a year**. A safety stock of 0.75 heat removes every stop but leaves only **0.4 %**. With a **3-heat buffer and a 0.5-heat safety stock**, the plan keeps **3.0 % with no caster stop**. |
| Two furnaces and a grid limit? | Two furnaces feeding 28 heats/day under an 80 MW contract (no simultaneous melting): the MILP stays easy (0.09 s, solved at the root node), and the limit costs **2.0 %** of the power bill. |

The decision this supports: the money is not in a smarter optimiser (a threshold rule gets most of it) but in
**buffer capacity**. One more heat of holding capacity turns a 0.4 % robust saving into 3.0 %, roughly
0.85 MEUR/yr on this plant.

## Models

**MILP** (`src/eafsched/model.py`, scipy.optimize.milp / HiGHS): binary start variables per furnace and quarter-hour;
no overlap on a furnace; buffer low point (caster draws before the steel arrives) above the safety stock and buffer
after arrival below capacity, every slot; closing stock at least the opening stock; optional grid power cap; symmetry
breaking between identical furnaces. Inventory is kept in integer units of 1/96 heat so that any whole number of heats
per day gives an integer draw per slot.

**Exact DP** (`dp.py`): state = (buffer level, quarter-hours of the current heat still to run), vectorised over the
buffer; used to certify the MILP and for the large sweeps and the re-planning simulation.

**Heuristics** (`heuristics.py`): price-blind (melt whenever the furnace is free and the steel fits) and a price
threshold with forcing rules (start if waiting one more slot would run the buffer dry or leave too little time to
close the day).

**Delays** (`sim.py`): each heat melts one extra quarter-hour with probability 0.15. Policies: follow the day-ahead plan
(late heats start as soon as the furnace is free), or re-optimise the rest of the day whenever the furnace becomes
free. Steel not melted by midnight is charged at the day's mean price, so a policy cannot look cheaper by melting less.

## Verification (`pytest tests`, 7 tests)

| Check | Result |
|---|---|
| DP vs brute-force enumeration of every schedule (5 random 24-slot days) | equal to 1e-9 |
| MILP vs DP (random days; all 1,090 real days in `run_all.py`) | 1e-6 in tests; 1.5e-10 EUR on real days |
| Independent feasibility check of MILP, DP and heuristic schedules | all feasible |
| LP relaxation is a lower bound | yes (1.3 % median gap) |
| Constant price: cost = heats x 40.5 MWh x price | exact |
| Two furnaces with an 80 MW cap: cap respected | peak 55.3 MW |
| Simulation without delays reproduces the plan's cost, no stops | exact |

## Limitations

* The plant is illustrative; real melt shops have variable heat energy, scrap mix, ladle-furnace and caster
  sequence constraints, and a separate value on caster interruptions (not priced here, only counted).
* Day-ahead prices are taken as known when the day is planned (they are, at noon the day before); intraday and
  balancing exposure, grid tariffs and demand charges are not modelled.
* Delays are independent; real ones cluster (electrode breaks, scrap quality).
* The re-planning policy gives the same decisions as following the plan when a safety stock is held: once the plan
  keeps a margin, starting a late heat immediately is already optimal.

## Run

    pip install numpy scipy pandas matplotlib pytest
    python scripts/download_prices.py      # IT-North 2023-2025 from api.energy-charts.info into data/
    python scripts/run_all.py 6            # about 15 min on 6 cores
    python scripts/buffer_delay.py         # buffer x safety stock under delays, about 5 min
    pytest tests

Results: `results/summary.json`, `results/buffer_delay.json`, `results/daily.csv`. Figures: `figures/`.
