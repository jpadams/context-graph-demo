#!/usr/bin/env python3
"""
Plant labeled credit-increase scenarios where the *text* decides the outcome.

Each scenario is a new synthetic customer who passes every code-checked rule (credit
score, income cap, increase cap, verified income, no sanctioned or poorly rated parties,
no frozen accounts, no open fraud case by status). The only thing that differs between
scenarios is narrative evidence that code cannot read:

                                                                          route     final
  control_clean                  nothing extra                               approve   approve
  ticket_resolved_genuine        closed dispute, customer confirms refund    approve   approve
  ticket_closed_still_disputed   closed dispute, customer says NOT resolved  escalate  reject (defer)
  condition_met                  conditional approval + follow-up met        approve   approve
  condition_unmet                conditional approval, no follow-up          escalate  reject (defer)
  alert_cleared_benign           cleared alert with documented explanation   approve   approve
  alert_closed_unexplained       alert bulk-closed, structuring unexplained  escalate  reject

Expected finals follow the bank's written escalation policy
(``decision_policy.ESCALATION_POLICY``, version 2026-11-11).

Every customer also gets one routine ticket, so there is always text to read.
Expected routes go to ``evals/labeled_cases.json`` for ``eval_decision_paths.py --labeled``.

Planted nodes carry no marker property (Claude's own Cypher tools would see it); they
have deterministic uuid5 ids, so re-running deletes and recreates them.

    cd backend && set -a && source ../.env && set +a
    .venv/bin/python scripts/plant_decision_scenarios.py [--per-scenario 2] [--remove]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import narratives  # noqa: E402

from app.context_graph_client import context_graph_client  # noqa: E402
from app.vector_client import vector_client  # noqa: E402

LABELS_PATH = Path(__file__).resolve().parents[1] / "evals" / "labeled_cases.json"
NS = uuid.UUID("6f1c2a52-9a43-4c41-9a5e-3b8f0d7f2a10")
NOW = datetime(2026, 11, 1)

SCENARIOS = {
    "control_clean": "approve",
    "ticket_resolved_genuine": "approve",
    "ticket_closed_still_disputed": "escalate",
    "condition_met": "approve",
    "condition_unmet": "escalate",
    "alert_cleared_benign": "approve",
    "alert_closed_unexplained": "escalate",
}
# Expected final answer under the written escalation policy (defer is reported as reject).
EXPECTED_FINAL = {
    "control_clean": "approve",
    "ticket_resolved_genuine": "approve",
    "ticket_closed_still_disputed": "reject",
    "condition_met": "approve",
    "condition_unmet": "reject",
    "alert_cleared_benign": "approve",
    "alert_closed_unexplained": "reject",
}
CONCERN_TYPES = ["transaction_dispute", "fraud_report", "complaint"]


def _id(*parts: object) -> str:
    return str(uuid.uuid5(NS, ":".join(map(str, parts))))


def _num(*parts: object) -> int:
    """Deterministic 6-digit number (``hash()`` is salted per process)."""
    return int(_id(*parts)[:8], 16) % 1_000_000


def _write(query: str, params: dict | None = None) -> list[dict]:
    """Write query (``execute_cypher`` is read-only by design)."""
    with context_graph_client.driver.session(database=context_graph_client.database) as session:
        return session.run(query, params or {}).data()


NODE_KINDS = [
    "person", "checking", "savings", "routine", "resolved_concern", "closed_but_unresolved",
    "conditional_approval", "condition_follow_up", "cleared_benign", "closed_but_unexplained",
]


def remove_planted(max_per_scenario: int = 20) -> int:
    """
    Delete planted nodes by their deterministic ids. Planted nodes carry no marker
    property: anything on a node is visible to Claude's own Cypher tools.
    """
    ids = [_id(sc, k, kind) for sc in SCENARIOS for k in range(max_per_scenario) for kind in NODE_KINDS]
    rows = _write("MATCH (n) WHERE n.id IN $ids DETACH DELETE n RETURN count(*) AS n", {"ids": ids})
    # Earlier versions tagged planted nodes; clear those too.
    legacy = _write(
        "MATCH (n) WHERE n.planted_scenario IS NOT NULL DETACH DELETE n RETURN count(*) AS n"
    )
    return (rows[0]["n"] if rows else 0) + (legacy[0]["n"] if legacy else 0)


def _existing_names() -> set[str]:
    return {r["n"] for r in context_graph_client.execute_cypher("MATCH (p:Person) RETURN p.normalized_name AS n")}


def plant_customer(scenario: str, k: int, name: str, rng: random.Random) -> dict:
    pid = _id(scenario, k, "person")
    income = rng.randrange(85_000, 140_000, 500)
    current = rng.randrange(12_000, 20_000, 500)
    amount = float(rng.choice([4_000, 5_000, 6_000]))
    _write(
        """
        CREATE (p:Person {
            id: $pid, name: $name, normalized_name: toLower($name), risk_score: $risk,
            credit_score: $score, annual_income: $income, income_verified: true,
            employment_status: 'employed', current_credit_limit: $current,
            credit_utilization: $util,created_at: datetime()
        })
        WITH p
        UNWIND $accounts AS acc
        CREATE (a:Account {
            id: acc.id, account_number: acc.number, account_type: acc.type, status: 'active',
            balance: acc.balance, currency: 'USD', risk_tier: 'low', opened_date: date($opened),
            source_system: 'Core Banking',created_at: datetime()
        })
        CREATE (p)-[:OWNS]->(a)
        """,
        {
            "pid": pid,
            "name": name,
            "risk": round(rng.uniform(0.12, 0.35), 3),
            "score": rng.randint(725, 800),
            "income": income,
            "current": current,
            "util": round(rng.uniform(0.15, 0.35), 2),
            "opened": (NOW - timedelta(days=rng.randint(900, 2500))).date().isoformat(),
            "accounts": [
                {
                    "id": _id(scenario, k, "checking"),
                    "number": f"ACC9{_num(scenario, k, 'checking') % 10_000_000:07d}",
                    "type": "checking",
                    "balance": round(rng.uniform(2_000, 15_000), 2),
                },
                {
                    "id": _id(scenario, k, "savings"),
                    "number": f"ACC9{_num(scenario, k, 'savings') % 10_000_000:07d}",
                    "type": "savings",
                    "balance": round(rng.uniform(5_000, 40_000), 2),
                },
            ],
        },
    )
    checking = _id(scenario, k, "checking")
    planted: list[dict] = []

    def ticket(kind: str, ticket_type: str, status: str, subject: str, text: str, days_ago: int) -> None:
        number = f"TKT9{_num(scenario, k, kind) % 100_000:05d}"
        _write(
            """
            MATCH (p:Person {id: $pid}), (a:Account {id: $acc})
            CREATE (t:SupportTicket {
                id: $id, ticket_number: $number, ticket_type: $type, priority: 'medium',
                status: $status, subject: $subject, description: $text, channel: 'phone',
                submitted_at: datetime($submitted), resolved_at: datetime($resolved),
                source_system: 'Support',created_at: datetime()
            })
            CREATE (t)-[:SUBMITTED_BY]->(p)
            CREATE (t)-[:REGARDING]->(a)
            """,
            {
                "pid": pid,
                "acc": checking,
                "id": _id(scenario, k, kind),
                "number": number,
                "type": ticket_type,
                "status": status,
                "subject": subject,
                "text": text,
                "submitted": (NOW - timedelta(days=days_ago)).isoformat(),
                "resolved": (NOW - timedelta(days=days_ago - 12)).isoformat(),
            },
        )
        planted.append({"kind": "ticket", "item_id": number, "variant": kind})

    def decision(kind: str, decision_type: str, status: str, reasoning: str, days_ago: int) -> None:
        did = _id(scenario, k, kind)
        _write(
            """
            MATCH (p:Person {id: $pid})
            CREATE (d:Decision {
                id: $id, decision_type: $type, category: 'credit', status: $status,
                decision_timestamp: datetime($ts), reasoning: $reasoning,
                reasoning_summary: $summary, confidence_score: 0.9, risk_factors: [],
                source_system: 'CRM',created_at: datetime()
            })
            CREATE (d)-[:ABOUT]->(p)
            """,
            {
                "pid": pid,
                "id": did,
                "type": decision_type,
                "status": status,
                "ts": (NOW - timedelta(days=days_ago)).isoformat(),
                "reasoning": reasoning,
                "summary": narratives.summary(reasoning),
            },
        )
        planted.append({"kind": "decision", "item_id": did, "variant": kind})

    def alert(kind: str, alert_type: str, status: str, text: str, days_ago: int) -> None:
        number = f"ALT9{_num(scenario, k, kind) % 100_000:05d}"
        _write(
            """
            MATCH (a:Account {id: $acc})
            CREATE (al:Alert {
                id: $id, alert_number: $number, alert_type: $type, severity: 'medium',
                status: $status, description: $text, triggered_at: datetime($ts),
                resolved_at: datetime($resolved), risk_score: 0.6, source_system: 'AML',
created_at: datetime()
            })
            CREATE (al)-[:REGARDING]->(a)
            """,
            {
                "acc": checking,
                "id": _id(scenario, k, kind),
                "number": number,
                "type": alert_type,
                "status": status,
                "text": text,
                "ts": (NOW - timedelta(days=days_ago)).isoformat(),
                "resolved": (NOW - timedelta(days=days_ago - 20)).isoformat(),
            },
        )
        planted.append({"kind": "alert", "item_id": number, "variant": kind})

    # Routine noise for everyone.
    subject, text = narratives._routine("statement_request", "resolved", rng)
    ticket("routine", "statement_request", "resolved", subject, text, rng.randint(40, 200))

    concern_type = CONCERN_TYPES[k % len(CONCERN_TYPES)]
    if scenario == "ticket_resolved_genuine":
        subject, text = narratives._concern_resolved(concern_type, rng)
        ticket("resolved_concern", concern_type, "resolved", subject, text, 60)
    elif scenario == "ticket_closed_still_disputed":
        subject, text = narratives._concern_closed_but_unresolved(concern_type, rng)
        ticket("closed_but_unresolved", concern_type, "resolved", subject, text, 45)
    elif scenario in ("condition_met", "condition_unmet"):
        reasoning, follow_up = narratives.conditional_credit_decision(_id(scenario, k, "cond"), current)
        decision("conditional_approval", "approval", "approved", reasoning, 240)
        if scenario == "condition_met":
            decision("condition_follow_up", "review", "completed", follow_up, 190)
    elif scenario == "alert_cleared_benign":
        alert_type = ["geographic_anomaly", "amount_threshold", "behavioral_anomaly"][k % 3]
        text = f"{alert_type.replace('_', ' ').title()}: {narratives._alert_benign(alert_type, rng)}"
        alert("cleared_benign", alert_type, "false_positive", text, 120)
    elif scenario == "alert_closed_unexplained":
        alert_type = ["pattern_match", "amount_threshold", "behavioral_anomaly"][k % 3]
        text = (
            f"{alert_type.replace('_', ' ').title()}: Closed as false positive by bulk closure rule "
            f"during backlog clean-up. Original finding: {narratives._alert_suspicious(alert_type, rng)} "
            "No analyst notes."
        )
        alert("closed_but_unexplained", alert_type, "false_positive", text, 90)

    return {
        "scenario": scenario,
        "expected_route": SCENARIOS[scenario],
        "expected_final": EXPECTED_FINAL[scenario],
        "policy_version": "2026-11-11",
        "customer": name,
        "customer_id": pid,
        "amount": amount,
        "planted": planted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-scenario", type=int, default=2)
    ap.add_argument("--remove", action="store_true", help="only remove planted nodes")
    args = ap.parse_args()

    removed = remove_planted()
    print(f"Removed {removed} previously planted nodes.")
    if args.remove:
        return

    fake = Faker()
    fake.seed_instance(2026)
    taken = _existing_names()
    cases = []
    for scenario in SCENARIOS:
        for k in range(args.per_scenario):
            name = fake.name()
            while name.lower() in taken or len(name.split()) != 2:
                name = fake.name()
            taken.add(name.lower())
            rng = random.Random(f"{scenario}:{k}")
            cases.append(plant_customer(scenario, k, name, rng))
            print(f"  {scenario:30s} {name:22s} ${cases[-1]['amount']:,.0f} -> {SCENARIOS[scenario]}")

    embedded = vector_client.batch_update_decision_embeddings(limit=500)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps({"generated": NOW.isoformat(), "cases": cases}, indent=1))
    print(f"Planted {len(cases)} customers; embedded {embedded} decisions. Labels: {LABELS_PATH}")


if __name__ == "__main__":
    main()
