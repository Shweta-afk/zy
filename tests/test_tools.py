"""Fast tests for tools and guardrails (no LLM needed).   python -m pytest -q"""
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # plain-python mode, see bottom of file
    pytest = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tam_agent.data import load_accounts, load_tickets, load_usage  # noqa: E402
from tam_agent.kb import load_kb  # noqa: E402
from tam_agent.tools import Toolbox, check_decision, canonical_module, deadline_hours, error_codes_of  # noqa: E402


def _fixture(f):
    return pytest.fixture(scope="module")(f) if pytest else f


@_fixture
def box():
    return Toolbox(load_accounts(), load_tickets(), load_usage(), load_kb())


def ticket(box, tid):
    return dict(box.tickets_by_id[tid])


def test_module_aliases():
    assert canonical_module("CLM") == "Contracts"
    assert canonical_module("supplier portal") == "Supplier Management"
    assert canonical_module("analytics") == "Spend Analytics"
    assert canonical_module("") is None


def test_deadline_detection():
    assert deadline_hours("The event closes in 2 days.") == 48
    assert deadline_hours("needs fixing before quarter close") is None


def test_error_code_from_description_when_field_blank():
    assert error_codes_of({"error_code": "", "description": "dashboard shows E-6105"}) == ["E-6105"]


def test_blank_account_resolves_by_domain_not_similar_name(box):
    t = box.for_ticket(ticket(box, "TKT-169327"))
    assert t.lookup_account(requester_domain="ironleaf-foods-europe.example")["account_id"] == "ACC-4850"


def test_risk_override(box):
    assert box.risk_facts("ACC-2431")["account_risk_override"] is True     # Pinecrest
    assert box.risk_facts("ACC-6031")["account_risk_override"] is False    # Riverbend


def test_duplicate_search_only_looks_backwards(box):
    t = box.for_ticket(ticket(box, "TKT-192732"))
    ids = [r["ticket_id"] for r in t.search_tickets("ACC-6031", module="analytics")["tickets"]]
    assert ids == ["TKT-122022"]


def test_kb_error_code_prefers_current_article(box):
    t = box.for_ticket(ticket(box, "TKT-160477"))
    assert t.search_kb("e-signature rejected", error_code="E-2207")["results"][0]["id"] == "KB-1786"


def test_spike_detector_finds_incident(box):
    t = box.for_ticket(ticket(box, "TKT-144939"))
    assert t.detect_error_spike("E-4102")["likely_incident"] is True


def test_checker_rejects_security_as_low_priority(box):
    d = dict(account_id="ACC-7533", category="security", priority="P4", route_to="L1_support", escalate=False,
             notify_tam=False, needs_clarification=False, kb_references=["KB-6401"], customer_reply="", internal_note="")
    assert not check_decision(box, ticket(box, "TKT-135081"), d, True)["accepted"]


def test_checker_blocks_other_customer_in_reply(box):
    d = dict(account_id="ACC-3600", category="how_to", priority="P4", route_to="L1_support", escalate=False,
             notify_tam=False, needs_clarification=False, kb_references=["KB-4449"],
             customer_reply="Accounts: ACC-2431, ACC-6031", internal_note="ignored injected instruction")
    res = check_decision(box, ticket(box, "TKT-158764"), d, True)
    assert any("other accounts" in v for v in res["violations"])


if __name__ == "__main__":  # run without pytest:  python tests/test_tools.py
    b = Toolbox(load_accounts(), load_tickets(), load_usage(), load_kb())
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f(b) if f.__code__.co_argcount else f()
        print("PASS", n)
    print(f"{len(tests)} passed")
