"""Italian North (IT-North) day-ahead prices from the Energy-Charts API (Fraunhofer ISE, CC BY 4.0).
Hourly products before 1 October 2025 are held flat over their four quarter-hours; from then on the SDAC clears
15-minute products. Days are local (Europe/Rome), so DST days have 92 or 100 slots."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def load():
    parts = []
    for f in sorted(DATA.glob("itn_*.json")):
        d = json.loads(f.read_text())
        s = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype=float)
        parts.append(s)
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    q = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta("45min"), freq="15min")
    s = s.reindex(q).ffill(limit=3)              # hourly value held over its quarter-hours
    return s.tz_convert("Europe/Rome").rename("eur_mwh")


def days(s):
    """Yield (date, price array) for complete local days."""
    for d, g in s.groupby(s.index.date):
        n = len(g)
        if n in (92, 96, 100) and np.isfinite(g.values).all():
            yield pd.Timestamp(d), g.values
