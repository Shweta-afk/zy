"""Score outputs/ against hand-labelled gold answers (data/gold_review.csv).

Labels were written by hand from the policy articles and the data, so treat them as one
engineer's judgement, not ground truth.   python scripts/eval_review.py
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["account_id", "category", "priority", "route_to", "escalate", "notify_tam", "duplicate_of",
          "needs_clarification"]


def norm(v):
    if isinstance(v, bool): return str(v).lower()
    return "" if v is None else str(v)


gold = list(csv.DictReader(open(ROOT / "data" / "gold_review.csv")))
hits = {f: 0 for f in FIELDS + ["kb_cited", "all_fields"]}
n = 0
print(f"{'ticket':12} " + " ".join(f[:8].ljust(8) for f in FIELDS) + " kb_cited")
for g in gold:
    p = ROOT / "outputs" / f"{g['ticket_id']}.json"
    if not p.exists():
        print(f"{g['ticket_id']:12} MISSING"); continue
    o = json.loads(p.read_text()); n += 1
    row, ok_all = [], True
    for f in FIELDS:
        ok = norm(o.get(f)) == g[f]
        hits[f] += ok; ok_all &= ok
        row.append(("ok" if ok else f"x:{norm(o.get(f))}")[:8].ljust(8))
    cited = not g["must_cite"] or g["must_cite"] in o.get("kb_references", [])
    hits["kb_cited"] += cited; hits["all_fields"] += ok_all
    print(f"{g['ticket_id']:12} " + " ".join(row) + (" ok" if cited else " MISSING " + g["must_cite"]))
print("\nAccuracy over", n, "tickets")
for f, h in hits.items():
    print(f"  {f:20} {h}/{n}  {h / max(n, 1):.0%}")
sec = [g for g in gold if g["category"] == "security"]
sec_ok = sum(json.loads((ROOT / "outputs" / f"{g['ticket_id']}.json").read_text())["category"] == "security"
             for g in sec if (ROOT / "outputs" / f"{g['ticket_id']}.json").exists())
print(f"  security recall      {sec_ok}/{len(sec)}")
