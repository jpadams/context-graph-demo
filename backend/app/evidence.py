"""
Deterministic evidence collection for approve/deny decisions.

A fixed set of Cypher retrievals runs in code (no tool-use LLM in the loop).
The same evidence is sent to Jev for a fast typed verdict and then to Claude
for the explanation, so both answer from identical context.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from .context_graph_client import context_graph_client
from .jev_client import jev_client
from .vector_client import vector_client

logger = logging.getLogger(__name__)

# Below this probability the message is not treated as an approval request.
APPROVAL_INTENT_THRESHOLD = 0.5

# Properties that only add noise (or size) to the state passed to Jev / Claude.
_EXCLUDED_PROPERTIES = {
    "fast_rp_embedding",
    "reasoning_embedding",
    "embedding",
    "description_embedding",
    "normalized_name",
    "created_at",
    "source_systems",
    "phone",
    "email",
    "date_of_birth",
}

_AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|K|m|M|thousand|million)?")
_NAME_STOPWORDS = {
    "should",
    "we",
    "approve",
    "credit",
    "limit",
    "increase",
    "for",
    "she",
    "he",
    "the",
    "find",
    "search",
    "customer",
    "override",
    "trading",
    "precedents",
    "similar",
    "exceptions",
    "check",
    "show",
    "me",
    "everything",
    "about",
    "requesting",
    "wants",
    "analyze",
    "his",
    "her",
}


def slim(record: Any, max_text: int = 600) -> Any:
    """Recursively drop embedding/noise properties and trim long strings."""
    if isinstance(record, dict):
        return {
            key: slim(value, max_text)
            for key, value in record.items()
            if key not in _EXCLUDED_PROPERTIES and value is not None
        }
    if isinstance(record, list):
        return [slim(item, max_text) for item in record]
    if isinstance(record, str) and len(record) > max_text:
        return record[:max_text] + "..."
    if hasattr(record, "isoformat"):
        return record.isoformat()
    return record


# ============================================
# MESSAGE PARSING (deterministic)
# ============================================


def extract_amount(message: str) -> float | None:
    """Pull the first dollar amount out of free text ("$25,000", "$25k", "$1.5M")."""
    match = _AMOUNT_RE.search(message)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    suffix = (match.group(2) or "").lower()
    if suffix in ("k", "thousand"):
        value *= 1_000
    elif suffix in ("m", "million"):
        value *= 1_000_000
    return value


def candidate_names(message: str) -> list[str]:
    """
    Candidate person names: runs of 2-3 consecutive capitalised words.
    Deterministic and cheap; the graph decides which candidates are real.
    """
    words = re.findall(r"[A-Z][a-z]+(?:['-][A-Za-z]+)?", message)
    tokens = [w for w in words if w.lower() not in _NAME_STOPWORDS]
    candidates: list[str] = []
    # Preserve adjacency by scanning the original text for consecutive capitalised words.
    runs = re.findall(
        r"(?:[A-Z][a-z]+(?:['-][A-Za-z]+)?\s+){1,2}[A-Z][a-z]+(?:['-][A-Za-z]+)?", message
    )
    for run in runs:
        parts = [p for p in run.split() if p.lower() not in _NAME_STOPWORDS]
        if len(parts) >= 2:
            candidates.append(" ".join(parts))
            if len(parts) == 3:
                candidates.append(" ".join(parts[:2]))
                candidates.append(" ".join(parts[1:]))
    # Fallback: any pair of remaining tokens
    if not candidates and len(tokens) >= 2:
        candidates.append(" ".join(tokens[:2]))
    seen: set[str] = set()
    ordered = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            ordered.append(c)
    return ordered


def resolve_customers(message: str) -> list[dict[str, Any]]:
    """Match candidate names in the message to Person nodes by normalized_name."""
    names = [c.lower() for c in candidate_names(message)]
    if not names:
        return []
    rows = context_graph_client.execute_cypher(
        """
        MATCH (p:Person)
        WHERE p.normalized_name IN $names
        RETURN p.id AS id, p.name AS name, p.risk_score AS risk_score
        ORDER BY size(p.name) DESC
        """,
        {"names": names},
    )
    return rows


# ============================================
# EVIDENCE RETRIEVALS (fixed Cypher set)
# ============================================

# Each retrieval is named so the UI can show exactly what was collected up front.
EVIDENCE_RETRIEVALS = [
    ("customer", "Customer profile with owned accounts and employer"),
    (
        "account_parties",
        "Organizations co-owning or transacting with the customer's accounts, with sanctions status",
    ),
    (
        "notable_transactions",
        "Flagged, reversed, pending or elevated-risk transactions on the customer's accounts",
    ),
    ("transaction_summary", "Transaction counts, volume and average risk by status"),
    ("support_tickets", "Support tickets submitted by the customer or about their accounts"),
    ("alerts", "Alerts regarding the customer's accounts or triggered by their transactions"),
    ("decision_history", "Past decisions about this customer"),
    ("precedents", "Precedent decisions via reasoning-embedding similarity"),
    ("category_outcomes", "Approval vs rejection base rates in this category"),
    ("policies", "Policies for this category"),
]

NOTABLE_RISK_THRESHOLD = 0.3


def _account_parties(customer_id: str) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (p:Person {id: $customer_id})-[:OWNS]->(a:Account)<-[:OWNS_ACCOUNT]-(o:Organization)
        RETURN 'co-owner of account' AS role, a.account_number AS account_number, o.name AS name,
               o.type AS org_type, o.industry AS industry, o.country AS country,
               o.risk_rating AS risk_rating, o.sanctions_status AS sanctions_status,
               null AS transactions
        UNION ALL
        MATCH (p:Person {id: $customer_id})-[:OWNS]->(a:Account)
              <-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)-[:COUNTERPARTY]->(o:Organization)
        WITH a, o, count(t) AS tx
        RETURN 'transaction counterparty' AS role, a.account_number AS account_number, o.name AS name,
               o.type AS org_type, o.industry AS industry, o.country AS country,
               o.risk_rating AS risk_rating, o.sanctions_status AS sanctions_status,
               tx AS transactions
        """,
        {"customer_id": customer_id},
    )


def _notable_transactions(customer_id: str, limit: int = 10) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (p:Person {id: $customer_id})-[:OWNS]->(a:Account)
        MATCH (a)<-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)
        WHERE t.status IN ['flagged', 'reversed', 'pending'] OR t.risk_score >= $threshold
        RETURN t.id AS id, a.account_number AS account_number, a.account_type AS account_type,
               t.amount AS amount, t.status AS status, t.transaction_type AS transaction_type,
               t.risk_score AS risk_score, toString(t.timestamp) AS timestamp,
               t.description AS description
        ORDER BY t.timestamp DESC
        LIMIT $limit
        """,
        {"customer_id": customer_id, "limit": limit, "threshold": NOTABLE_RISK_THRESHOLD},
    )


def _transaction_summary(customer_id: str) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (p:Person {id: $customer_id})-[:OWNS]->(a:Account)
              <-[:FROM_ACCOUNT|TO_ACCOUNT]-(t:Transaction)
        RETURN t.status AS status, count(*) AS count, round(sum(t.amount), 2) AS total_amount,
               round(avg(t.risk_score), 3) AS avg_risk_score, max(t.amount) AS max_amount
        ORDER BY count DESC
        """,
        {"customer_id": customer_id},
    )


def _support_tickets(customer_id: str, limit: int = 10) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (p:Person {id: $customer_id})
        OPTIONAL MATCH (p)<-[:SUBMITTED_BY]-(t1:SupportTicket)
        OPTIONAL MATCH (p)-[:OWNS]->(:Account)<-[:REGARDING]-(t2:SupportTicket)
        WITH collect(DISTINCT t1) + collect(DISTINCT t2) AS tickets
        UNWIND tickets AS t
        RETURN DISTINCT t.ticket_number AS ticket_number, t.ticket_type AS ticket_type,
               t.priority AS priority, t.status AS status, t.subject AS subject,
               toString(t.submitted_at) AS submitted_at, toString(t.resolved_at) AS resolved_at
        ORDER BY submitted_at DESC
        LIMIT $limit
        """,
        {"customer_id": customer_id, "limit": limit},
    )


def _alerts(customer_id: str, limit: int = 10) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (p:Person {id: $customer_id})-[:OWNS]->(a:Account)
        OPTIONAL MATCH (a)<-[:REGARDING]-(al1:Alert)
        OPTIONAL MATCH (a)<-[:FROM_ACCOUNT|TO_ACCOUNT]-(:Transaction)<-[:TRIGGERED_BY]-(al2:Alert)
        WITH collect(DISTINCT al1) + collect(DISTINCT al2) AS alerts
        UNWIND alerts AS al
        RETURN DISTINCT al.alert_number AS alert_number, al.alert_type AS alert_type,
               al.severity AS severity, al.status AS status, al.description AS description,
               al.risk_score AS risk_score, toString(al.triggered_at) AS triggered_at
        ORDER BY triggered_at DESC
        LIMIT $limit
        """,
        {"customer_id": customer_id, "limit": limit},
    )


def _category_outcomes(category: str | None) -> list[dict]:
    return context_graph_client.execute_cypher(
        """
        MATCH (d:Decision)
        WHERE $category IS NULL OR d.category = $category
        RETURN d.decision_type AS decision_type, d.status AS status, count(*) AS count
        ORDER BY count DESC
        """,
        {"category": category},
    )


def collect_evidence(
    customer_id: str, request_summary: str, category: str | None
) -> dict[str, Any]:
    """Run the fixed retrieval set. Returns evidence plus what was run and timings."""
    evidence: dict[str, Any] = {}
    retrievals: list[dict[str, Any]] = []

    def run(name: str, description: str, fn):
        started = time.perf_counter()
        try:
            result = fn()
            status = "ok"
        except Exception as e:  # keep going; the verdict can note missing evidence
            logger.warning(f"Evidence retrieval '{name}' failed: {e}")
            result, status = [], f"failed: {e}"
        evidence[name] = slim(result)
        count = len(result) if isinstance(result, list) else (1 if result else 0)
        retrievals.append(
            {
                "name": name,
                "description": description,
                "count": count,
                "status": status,
                "ms": round((time.perf_counter() - started) * 1000),
            }
        )

    descriptions = dict(EVIDENCE_RETRIEVALS)
    run(
        "customer", descriptions["customer"], lambda: context_graph_client.get_customer(customer_id)
    )
    run("account_parties", descriptions["account_parties"], lambda: _account_parties(customer_id))
    run(
        "notable_transactions",
        descriptions["notable_transactions"],
        lambda: _notable_transactions(customer_id),
    )
    run(
        "transaction_summary",
        descriptions["transaction_summary"],
        lambda: _transaction_summary(customer_id),
    )
    run("support_tickets", descriptions["support_tickets"], lambda: _support_tickets(customer_id))
    run("alerts", descriptions["alerts"], lambda: _alerts(customer_id))
    run(
        "decision_history",
        descriptions["decision_history"],
        lambda: context_graph_client.get_customer_decisions(customer_id=customer_id, limit=10),
    )
    run(
        "precedents",
        descriptions["precedents"],
        lambda: vector_client.find_precedents_hybrid(
            scenario=request_summary, category=category, limit=5
        ),
    )
    run(
        "category_outcomes", descriptions["category_outcomes"], lambda: _category_outcomes(category)
    )
    run(
        "policies",
        descriptions["policies"],
        lambda: context_graph_client.get_policies(category=category),
    )

    return {"evidence": evidence, "retrievals": retrievals}


def derive_credit_figures(request: dict[str, Any], customer: Optional[dict[str, Any]]) -> dict:
    """
    Arithmetic the policy needs, done in code: proposed total limit, its share of income,
    and the increase relative to the current limit. Only present when the inputs exist.
    """
    amount = request.get("requested_amount_usd")
    if not customer or amount is None:
        return {}
    derived: dict[str, Any] = {}
    current = customer.get("current_credit_limit")
    income = customer.get("annual_income")
    if current is not None:
        derived["current_credit_limit_usd"] = current
        derived["proposed_total_limit_usd"] = current + amount
        if current > 0:
            derived["increase_pct_of_current_limit"] = round(amount / current, 3)
    if income:
        total = derived.get("proposed_total_limit_usd", amount)
        derived["proposed_total_limit_pct_of_income"] = round(total / income, 3)
    return derived


def build_decision_state(request: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """The JSON state Jev evaluates: named fields keep relationships clear."""
    request = dict(request)
    derived = derive_credit_figures(request, evidence.get("customer"))
    if derived:
        request["derived"] = derived
    return {
        "request": slim(request),
        "customer": evidence.get("customer") or {"note": "No customer record found"},
        "account_parties": evidence.get("account_parties", []),
        "notable_transactions": evidence.get("notable_transactions", []),
        "transaction_summary": evidence.get("transaction_summary", []),
        "support_tickets": evidence.get("support_tickets", []),
        "alerts": evidence.get("alerts", []),
        "decision_history": evidence.get("decision_history", []),
        "precedents": evidence.get("precedents", []),
        "category_outcomes": evidence.get("category_outcomes", []),
        "policies": evidence.get("policies", []),
    }


def customer_graph_data(customer_id: str, depth: int = 2, limit: int = 30) -> dict[str, Any]:
    """Graph visualization data centred on the customer, for the UI."""
    try:
        graph = context_graph_client.get_graph_data(
            center_node_id=customer_id, depth=depth, limit=limit
        )
        nodes = [
            {"id": n.id, "labels": n.labels, "properties": slim(n.properties, max_text=200)}
            for n in graph.nodes
        ]
        node_ids = {n["id"] for n in nodes}
        relationships = [
            {
                "id": r.id,
                "type": r.type,
                "startNodeId": r.start_node_id,
                "endNodeId": r.end_node_id,
                "properties": slim(r.properties, max_text=200),
            }
            for r in graph.relationships
            if r.start_node_id in node_ids and r.end_node_id in node_ids
        ]
        return {"nodes": nodes, "relationships": relationships}
    except Exception as e:
        logger.warning(f"Graph data for customer {customer_id} unavailable: {e}")
        return {"nodes": [], "relationships": []}


# ============================================
# FAST ANSWER PIPELINE
# ============================================


def prepare_fast_answer(message: str) -> dict[str, Any] | None:
    """
    Fast path for approve/deny questions.

    1. Jev reads the message: is it an approval request, which category and type?
    2. Code resolves the customer named in the message against the graph.
    3. Code runs the fixed evidence retrievals.
    4. Jev returns the typed verdict over that evidence.

    Returns None when the fast path does not apply (not an approval request,
    no customer resolved, or Jev not configured).
    """
    if not jev_client.is_configured:
        return None

    started = time.perf_counter()
    intent = jev_client.classify_request(message)
    if intent["is_approval_request"] < APPROVAL_INTENT_THRESHOLD:
        logger.info(f"Fast path skipped: not an approval request ({intent})")
        return {"applicable": False, "reason": "not_an_approval_request", "intent": intent}

    customers = resolve_customers(message)
    if not customers:
        logger.info("Fast path skipped: no customer resolved from message")
        return {"applicable": False, "reason": "no_customer_resolved", "intent": intent}
    customer = customers[0]

    from .cases import run_case  # local import: cases builds on this module

    category = None if intent["category"] == "other" else intent["category"]
    request = {
        "type": intent["request_type"],
        "category": category,
        "summary": message,
        "customer_name": customer["name"],
    }
    amount = extract_amount(message)
    if amount is not None:
        request["requested_amount_usd"] = amount

    result = run_case(customer, request, intent=intent)
    result["total_ms"] = round((time.perf_counter() - started) * 1000)
    return result
