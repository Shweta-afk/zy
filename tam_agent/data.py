"""Data loaders. Loading only - no joins, matching or business logic.

Pure standard library (csv), so the project runs on locked-down machines where compiled
packages such as pandas are blocked. Each loader returns a list of dicts (one per row).
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> list[dict]:
    with open(DATA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_accounts() -> list[dict]:
    return _read("accounts.csv")


def load_tickets() -> list[dict]:
    return _read("tickets.csv")


def load_usage() -> list[dict]:
    rows = _read("usage_metrics.csv")
    for r in rows:
        r["active_users"] = int(r["active_users"])
        r["documents_processed"] = int(r["documents_processed"])
        r["login_rate_pct"] = float(r["login_rate_pct"])
        r["tickets_opened"] = int(r["tickets_opened"])
    return rows


def load_review_set() -> list[str]:
    return [l.strip() for l in (DATA / "review_set.txt").read_text().splitlines() if l.strip()]


AS_OF_DATE = "2026-10-20"  # treat this as "today" for all date calculations
