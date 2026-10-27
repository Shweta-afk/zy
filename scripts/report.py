"""Build outputs/report.html: one card per triaged ticket with its decision and tool-call timeline.

    python scripts/report.py      (then open outputs/report.html in a browser)
"""
import html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
ids = [l.strip() for l in (ROOT / "data" / "review_set.txt").read_text().splitlines() if l.strip()]
ids += sorted(p.stem for p in OUT.glob("TKT-*.json") if p.stem not in ids)
tickets = {}
try:
    import csv
    tickets = {r["ticket_id"]: r for r in csv.DictReader(open(ROOT / "data" / "tickets.csv"))}
except Exception:
    pass

COLORS = {"P1": "#c62828", "P2": "#ef6c00", "P3": "#1565c0", "P4": "#546e7a"}
e = html.escape
cards = []
for tid in ids:
    p = OUT / f"{tid}.json"
    if not p.exists():
        continue
    o = json.loads(p.read_text())
    t = tickets.get(tid, {})
    chips = [f'<span class="chip" style="background:{COLORS.get(o["priority"], "#555")};color:#fff">{o["priority"]}</span>',
             f'<span class="chip">{e(o["category"])}</span>', f'<span class="chip">&rarr; {e(o["route_to"])}</span>']
    for k, label in [("escalate", "escalate"), ("notify_tam", "notify TAM"), ("needs_clarification", "needs info")]:
        if o.get(k): chips.append(f'<span class="chip warn">{label}</span>')
    if o.get("duplicate_of"): chips.append(f'<span class="chip warn">duplicate of {e(o["duplicate_of"])}</span>')
    steps = "".join(
        f'<li class="{ "rej" if "REJECTED" in s.get("observation", "") else ("fix" if s["action"] == "guardrail_autofix" else "") }">'
        f'<b>{s["step"]}. {e(s["action"])}</b> <code>{e(str(s.get("input", ""))[:160])}</code>'
        + (f'<div class="th">{e(s["thought"])}</div>' if s.get("thought") else "")
        + f'<div>{e(str(s["observation"]))}</div></li>' for s in o["reasoning_trace"])
    cards.append(f"""<section class="card"><h2>{tid} <small>{e(o.get('account_id') or 'unresolved')} &middot;
      confidence {o['confidence']}</small></h2><p class="subj">{e(t.get('subject', ''))}</p>
      <div class="chips">{''.join(chips)}</div>
      <p><b>KB:</b> {', '.join(o['kb_references']) or '-'} &nbsp; <b>Signals:</b> {e('; '.join(o['account_signals']))}</p>
      <p><b>Internal note:</b> {e(o['internal_note'])}</p>
      <details><summary>Customer reply</summary><p>{e(o['customer_reply'])}</p></details>
      <details open><summary>Reasoning trace ({len(o['reasoning_trace'])} steps)</summary><ol>{steps}</ol></details></section>""")

page = f"""<!doctype html><meta charset="utf-8"><title>Triage report</title><style>
body{{font:14px/1.5 system-ui,Segoe UI,Arial;margin:0;background:#f4f6f9;color:#1d2733}}
main{{max-width:980px;margin:auto;padding:24px 16px}} h1{{margin:0 0 4px}}
.card{{background:#fff;border:1px solid #d9e0e8;border-radius:8px;padding:16px 20px;margin:16px 0}}
h2{{margin:0;font-size:17px}} h2 small{{color:#5b6b7c;font-weight:400}} .subj{{color:#33475b;margin:4px 0 8px}}
.chip{{display:inline-block;padding:2px 9px;border-radius:12px;background:#e8eef5;margin:0 6px 6px 0;font-size:12px}}
.chip.warn{{background:#fff1d6}} ol{{padding-left:20px}} li{{margin:6px 0;padding:4px 8px;border-left:3px solid #c9d3df}}
li.rej{{border-color:#c62828;background:#fdf1f1}} li.fix{{border-color:#ef6c00;background:#fff7ec}}
code{{background:#f0f3f7;padding:1px 4px;border-radius:3px;font-size:12px}} .th{{color:#5b6b7c;font-style:italic}}
summary{{cursor:pointer;font-weight:600}}</style><main><h1>Ticket triage report</h1>
<p>{len(cards)} tickets. Red steps = submission rejected by the policy checker; orange = code guardrail fix.</p>
{''.join(cards)}</main>"""
(OUT / "report.html").write_text(page, encoding="utf-8")
print(f"Wrote {OUT / 'report.html'} ({len(cards)} tickets)")
