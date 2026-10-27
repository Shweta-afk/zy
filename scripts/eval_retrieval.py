"""Retrieval quality without any labelling and without the LLM.

KB front matter lists the error codes each article covers, so every ticket that carries an
error code has a known correct (current) article. We measure how often search_kb puts it in
the top 1 / top 5, with plain BM25 versus BM25 + the exact error-code index.

    python scripts/eval_retrieval.py
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.tools import Toolbox, canonical_module  # noqa: E402

box = Toolbox(load_accounts(), load_tickets(), load_usage(), load_kb())
t = box.tickets
cases = []
for r in t:
    for code in r.codes[:1]:
        good = {k for k in box.code_index.get(code, []) if box.kb[k]["meta"].get("status") != "deprecated"}
        if good:
            cases.append((f"{r.subject} {r.description}", r.module_canon, code, good))

import re  # noqa: E402
variants = [("A. BM25, symptom text only (code removed)", "strip"),
            ("B. BM25, text incl. error code", "bm25"),
            ("C. BM25 + exact error-code index (used)", "hybrid")]
for label, mode in variants:
    t0 = time.time(); h1 = h5 = 0
    for q, mod, code, good in cases:
        if mode == "strip":
            ids = [k for k, _ in box.search_kb_raw(re.sub(r"E-\d{4}", "", q), mod, None, k=5, use_code_index=False)]
        else:
            ids = [k for k, _ in box.search_kb_raw(q, mod, code, k=5, use_code_index=mode == "hybrid")]
        h1 += ids[0] in good; h5 += bool(good & set(ids))
    ms = (time.time() - t0) / len(cases) * 1000
    print(f"{label:44} n={len(cases)}  recall@1={h1 / len(cases):.1%}  recall@5={h5 / len(cases):.1%}  "
          f"{ms:.1f} ms/query")
no_code = sum(1 for r in t if not r.codes)
print(f"\n{no_code} tickets have no error code; those rely on BM25 + module filter only (not measured here).")
