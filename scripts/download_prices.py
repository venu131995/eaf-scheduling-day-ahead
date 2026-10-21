"""IT-North day-ahead prices 2023-2025 from the Energy-Charts API (Fraunhofer ISE, CC BY 4.0) into data/itn_*.json."""
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
URL = "https://api.energy-charts.info/price?bzn=IT-North&start={a}&end={b}"

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for y in (2023, 2024, 2025):
        for a, b in (("01-01", "06-30"), ("07-01", "12-31")):
            f = DATA / f"itn_{y}_{a}.json"
            if not f.exists():
                urllib.request.urlretrieve(URL.format(a=f"{y}-{a}", b=f"{y}-{b}"), f)
            print(f.name, f.stat().st_size)
