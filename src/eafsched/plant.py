"""Illustrative electric-arc-furnace (EAF) mini-mill, in quarter-hour slots.

A heat (charge) of 100 t needs 40 MWh of power-on energy (400 kWh/t, a typical EAF figure), drawn over 3 slots
(45 min at 53.3 MW), then 1 slot of tapping and charging at 2 MW: a 60-minute tap-to-tap cycle. Liquid steel goes
into a ladle buffer (ladle furnace / holding) from which the continuous caster draws a constant 20 heats per day
(2,000 t/day, about 0.73 Mt/yr). The buffer holds at most `buf_max` heats and must never run dry (caster stop).

Inventory is counted in integer units of 1/96 heat so that the caster draw per slot is an integer for any whole
number of heats per day: one heat = 96 units, a caster taking H heats per day draws H units per slot.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Plant:
    power_on: int = 3                 # slots of melting per heat
    tap: int = 1                      # slots of tapping/charging after melting
    mw_melt: float = 40.0 / 0.75      # MW during melting (40 MWh in 45 min)
    mw_tap: float = 2.0               # MW during tapping/charging
    unit: int = 96                    # inventory units per heat
    draw: int = 20                    # caster draw, units per slot (= heats per day on a 96-slot day)
    buf_max: int = 192                # buffer capacity in units (2 heats)
    buf_min: int = 0                  # safety stock in units
    n_furnaces: int = 1
    power_cap: float = field(default=1e9)   # grid contract limit on total MW (binding only with 2+ furnaces)

    @property
    def cycle(self):
        return self.power_on + self.tap

    def profile(self):
        """MW drawn in each slot of a heat, from its start slot."""
        return [self.mw_melt] * self.power_on + [self.mw_tap] * self.tap

    def energy_per_heat(self):
        return sum(self.profile()) * 0.25
