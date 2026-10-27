"""Print the reasoning trace of triaged tickets.   python scripts/show_trace.py TKT-160067 TKT-119877"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
for tid in sys.argv[1:] or sorted(p.stem for p in OUT.glob("TKT-*.json")):
    o = json.loads((OUT / f"{tid}.json").read_text())
    print("=" * 30, tid, o["category"], o["priority"], o["route_to"], "confidence", o["confidence"])
    for s in o["reasoning_trace"]:
        print(f'{s["step"]:>2}. {s["action"]:<22} {str(s.get("input", ""))[:70]}')
        print(f'      -> {str(s["observation"])[:260]}')
