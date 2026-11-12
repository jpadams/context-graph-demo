"""
Decision cases: parameterized standard operating procedures for AI decisions
on the context graph.

A case fixes three things so every evaluation runs the same way:

* **slots** - the parameters a case needs (customer, amount, ...);
* **an evidence plan** - the fixed set of Cypher retrievals run in code
  (see ``evidence.collect_evidence``), never left to tool selection;
* **judgments and policy** - by default ("decomposed") code computes the policy
  facts, Jev answers one narrow Noul per evidence item that needs reading, and
  ``decision_policy.combine`` decides with named guards
  (see ``decide``). The "monolithic" baseline asks one Jev Choice instead
  (``jev_client.evaluate_decision``).

The chat path fills the slots from the user's message; the API and eval
harness fill them directly, so the same case runs for customer_x / amount_y.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from .config import config
from .context_graph_client import context_graph_client
from .decision_policy import (
    ESCALATION_POLICY_VERSION,
    combine,
    compute_facts,
    escalation_policy_for,
    facts_as_list,
)
from .evidence import (
    build_decision_state,
    collect_evidence,
    customer_graph_data,
    derive_credit_figures,
)
from .jev_client import jev_client

logger = logging.getLogger(__name__)

CREDIT_LINE_INCREASE = "credit_limit_increase"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def resolve_customer(customer: str) -> dict[str, Any] | None:
    """Resolve a customer slot given as an id, an exact name, or a partial name."""
    customer = customer.strip()
    if _UUID_RE.match(customer):
        rows = context_graph_client.execute_cypher(
            "MATCH (p:Person {id: $id}) RETURN p.id AS id, p.name AS name, p.risk_score AS risk_score",
            {"id": customer},
        )
        if rows:
            return rows[0]
    rows = context_graph_client.execute_cypher(
        """
        MATCH (p:Person)
        WHERE p.normalized_name = toLower($name)
        RETURN p.id AS id, p.name AS name, p.risk_score AS risk_score
        LIMIT 1
        """,
        {"name": customer},
    )
    if rows:
        return rows[0]
    matches = context_graph_client.search_customers(query=customer, limit=1)
    if matches:
        m = matches[0]
        return {"id": m["id"], "name": m["name"], "risk_score": m.get("risk_score")}
    return None


def credit_line_increase_message(customer_name: str, amount: float) -> str:
    """The canonical natural-language form of the case (used for precedent search and Claude)."""
    return (
        f"Should we approve a credit limit increase for {customer_name}? "
        f"They are requesting a ${amount:,.0f} limit increase."
    )


def build_credit_line_increase_request(
    customer: dict[str, Any], amount: float, summary: str | None = None
) -> dict[str, Any]:
    return {
        "type": CREDIT_LINE_INCREASE,
        "category": "credit",
        "summary": summary or credit_line_increase_message(customer["name"], amount),
        "customer_name": customer["name"],
        "requested_amount_usd": float(amount),
    }


def collect_case_evidence(customer: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """
    The evidence half of a case: fixed retrieval plan plus code-computed policy
    arithmetic. No model involved. Returns the filled request, the evidence, the
    retrieval report and the state exactly as a decision model would receive it.
    """
    started = time.perf_counter()
    collected = collect_evidence(customer["id"], request["summary"], request.get("category"))
    derived = derive_credit_figures(request, collected["evidence"].get("customer"))
    if derived:  # shown to Jev, Claude and the UI alike
        request = {**request, "derived": derived}
    return {
        "customer": customer,
        "request": request,
        "retrievals": collected["retrievals"],
        "evidence": collected["evidence"],
        "state": build_decision_state(request, collected["evidence"]),
        "evidence_ms": round((time.perf_counter() - started) * 1000),
    }


def decide(state: dict[str, Any], mode: str | None = None) -> dict[str, Any]:
    """
    The verdict for a decision state.

    ``decomposed``: code facts + per-item Jev judgments -> guard combiner. The verdict
    carries the full trace (facts, judgments, guards, which guard decided).
    ``monolithic``: one Jev Choice decides (baseline).
    Both return ``decision`` / ``routed_decision`` in the yes / no / needs_llm_decision
    vocabulary used by the Claude hand-off, the UI and the eval.
    """
    mode = mode or config.typesafe.decision_mode
    if mode == "monolithic":
        return jev_client.evaluate_decision(state)

    facts = compute_facts(state)
    judged = jev_client.evaluate_judgments(state)
    thresholds = judged["thresholds"]
    result = combine(facts, judged["judgments"], thresholds["yes"], thresholds["no"])
    return {
        "mode": "decomposed",
        "model": judged["model"],
        "decision": result["routed_decision"],
        "routed_decision": result["routed_decision"],
        "outcome": result["outcome"],
        "decided_by": result["decided_by"],
        "fired": result["fired"],
        "guards": result["guards"],
        "escalation_policy": (
            {
                "version": ESCALATION_POLICY_VERSION,
                "entries": escalation_policy_for(result["fired"]),
            }
            if result["outcome"] == "escalate"
            else None
        ),
        "facts": facts_as_list(facts),
        "judgments": judged["judgments"],
        "thresholds": thresholds,
        "question_count": judged["question_count"],
        "usage": judged["usage"],
        "latency_ms": judged["latency_ms"],
        "request_id": judged["request_id"],
    }


def run_case(
    customer: dict[str, Any],
    request: dict[str, Any],
    intent: dict | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Run a decision case for a resolved customer and a filled request:
    fixed evidence plan -> verdict (see ``decide``). Same output shape as the chat fast path.
    """
    started = time.perf_counter()
    case = collect_case_evidence(customer, request)
    verdict = decide(case["state"], mode)
    return {
        "applicable": True,
        "case": request["type"],
        "intent": intent,
        "customer": customer,
        "request": case["request"],
        "retrievals": case["retrievals"],
        "evidence": case["evidence"],
        "verdict": verdict,
        "graph_data": customer_graph_data(customer["id"]),
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


def collect_credit_line_increase_evidence(customer: str, amount: float) -> dict[str, Any]:
    """Evidence only (no Jev) for the parameterized case; used by the evidence-only eval arm."""
    resolved = resolve_customer(customer)
    if not resolved:
        return {"applicable": False, "reason": "no_customer_resolved", "customer_query": customer}
    request = build_credit_line_increase_request(resolved, amount)
    return {"applicable": True, **collect_case_evidence(resolved, request)}


def run_credit_line_increase(
    customer: str, amount: float, mode: str | None = None
) -> dict[str, Any]:
    """The Jessica Norris case for any customer_x and amount_y."""
    if not jev_client.is_configured:
        raise RuntimeError("TYPESAFE_API_KEY is not configured")
    resolved = resolve_customer(customer)
    if not resolved:
        return {"applicable": False, "reason": "no_customer_resolved", "customer_query": customer}
    request = build_credit_line_increase_request(resolved, amount)
    return run_case(resolved, request, mode=mode)
