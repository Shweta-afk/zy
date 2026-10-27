"""Show the spike detector's timeline for one ticket (no LLM or internet needed).

    python scripts/demo_spike.py                      # TKT-144939 / E-4102
    python scripts/demo_spike.py TKT-160477 E-2207
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.tools import Toolbox  # noqa: E402

box = Toolbox(load_accounts(), load_tickets(), load_usage(), load_kb())
tid, code = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("TKT-144939", "E-4102")
t = box.for_ticket(dict(box.tickets_by_id[tid]))
since = t.now - timedelta(hours=48)
print(f"{tid} created {t.now}  (window: 48h before it)")
for x in sorted((x for x in box.tickets if code in x.codes and since <= x.ts <= t.now + timedelta(hours=24)),
                key=lambda x: x.ts):
    tag = "THIS TICKET" if x.ticket_id == tid else ("BEFORE (counted)" if x.ts < t.now else "AFTER (ignored)")
    print(f"  {x.created_at}  {x.ticket_id}  {x.account_resolved}  {tag}")
print(t.detect_error_spike(code)["summary"])