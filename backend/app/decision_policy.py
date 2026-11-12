"""
Decomposed decision policy: code makes the decision, Jev supplies judgments.

"Jev makes judgments, code makes decisions." This module holds the code half:

* ``compute_facts`` - everything that is a lookup, a comparison or arithmetic over the
  collected evidence (policy thresholds, sanctions status, open fraud cases, approval
  tier, evidence completeness). No model involved.
* ``combine`` - an ordered list of named guards over those facts and Jev's per-item
  judgments. The first guard that fires decides; every guard is reported, so a wrong
  answer can be traced to either a wrong judgment or a wrong rule.

Two design rules:

1. **Only code facts can reject.** A Jev judgment can stop an automatic approval and
   escalate the case, but never deny credit on its own. The semantic read of a ticket is
   advisory; the denial must rest on a written rule.
2. **Unsure judgments escalate.** A judgment between the ``no`` and ``yes`` thresholds is
   reported as ``unsure`` and escalates, rather than being rounded to either side.

Pure functions only: no I/O, so the policy is unit-testable without Neo4j or Jev.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Defaults; the Credit Limit / Exception Approval Authority policy nodes override them
# through their ``threshold_rules`` JSON.
DEFAULT_CREDIT_RULES = {
    "min_credit_score": 660,
    "max_total_limit_pct_of_income": 0.35,
    "max_increase_pct_of_current_limit": 1.0,
    "senior_review_above_usd": 50000,
}
DEFAULT_AUTHORITY_RULES = {
    "manager_approval_above_usd": 25000,
    "senior_approval_above_usd": 50000,
    "committee_approval_above_usd": 100000,
}

# Bank settings, not model properties.
ELEVATED_RISK_SCORE = 0.7
POOR_COUNTERPARTY_RATINGS = {"D"}
OPEN_FRAUD_TICKET_TYPES = {"fraud_report"}
OPEN_TICKET_STATUSES = {"open", "in_progress", "pending_customer"}
OPEN_FRAUD_ALERT_TYPES = {"fraud_detection", "sanctions_hit"}
OPEN_ALERT_STATUSES = {"open", "investigating", "escalated"}
REQUIRED_CUSTOMER_FIELDS = ("credit_score", "annual_income", "current_credit_limit")

APPROVE, REJECT, ESCALATE = "approve", "reject", "escalate"
# The routed vocabulary the rest of the app (Claude hand-off, UI, eval) already uses.
ROUTED = {APPROVE: "yes", REJECT: "no", ESCALATE: "needs_llm_decision"}


@dataclass
class Fact:
    name: str
    value: Any  # True / False / None (unknown) or a label for informational facts
    detail: str
    label: str = ""
    source: str = "code"


@dataclass
class Guard:
    name: str
    effect: str  # reject | escalate
    source: str  # code | jev
    status: str  # fired | passed | not_applicable
    reason: str
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Facts (code)
# ---------------------------------------------------------------------------


def _rules(policies: list[dict], name: str, defaults: dict) -> dict:
    for p in policies or []:
        if p.get("name") == name and p.get("threshold_rules"):
            raw = p["threshold_rules"]
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
                return {**defaults, **parsed}
            except (ValueError, TypeError):
                break
    return dict(defaults)


def _usd(x: float | int | None) -> str:
    return "unknown" if x is None else f"${x:,.0f}"


def compute_facts(state: dict[str, Any]) -> dict[str, Fact]:
    """Every check that code can answer exactly, over the decision state."""
    request = state.get("request") or {}
    derived = request.get("derived") or {}
    customer = state.get("customer") or {}
    policies = state.get("policies") or []
    credit = _rules(policies, "Credit Limit Policy", DEFAULT_CREDIT_RULES)
    authority = _rules(policies, "Exception Approval Authority", DEFAULT_AUTHORITY_RULES)
    amount = request.get("requested_amount_usd")
    facts: dict[str, Fact] = {}

    def add(name: str, value: Any, detail: str, label: str) -> None:
        facts[name] = Fact(name=name, value=value, detail=detail, label=label)

    missing = [f for f in REQUIRED_CUSTOMER_FIELDS if customer.get(f) is None]
    if amount is None:
        missing.append("requested_amount_usd")
    add(
        "evidence_complete",
        not missing,
        "all required fields present" if not missing else "missing: " + ", ".join(missing),
        "Required fields present",
    )

    score = customer.get("credit_score")
    add(
        "credit_score_meets_minimum",
        None if score is None else score >= credit["min_credit_score"],
        f"credit score {score if score is not None else 'unknown'} vs minimum "
        f"{credit['min_credit_score']}",
        "Credit score meets minimum",
    )

    pct_income = derived.get("proposed_total_limit_pct_of_income")
    cap = credit["max_total_limit_pct_of_income"]
    add(
        "total_limit_within_income_cap",
        None if pct_income is None else pct_income <= cap,
        f"proposed total {_usd(derived.get('proposed_total_limit_usd'))} is "
        + ("unknown" if pct_income is None else f"{pct_income:.0%}")
        + f" of income (cap {cap:.0%})",
        "Total limit within income cap",
    )

    pct_current = derived.get("increase_pct_of_current_limit")
    inc_cap = credit["max_increase_pct_of_current_limit"]
    add(
        "increase_within_current_limit_cap",
        None if pct_current is None else pct_current <= inc_cap,
        f"increase is "
        + ("unknown" if pct_current is None else f"{pct_current:.0%}")
        + f" of the current limit (cap {inc_cap:.0%})",
        "Increase within current-limit cap",
    )

    verified = customer.get("income_verified")
    add(
        "income_verified",
        None if verified is None else bool(verified),
        f"income_verified = {verified}",
        "Income verified",
    )

    add(
        "senior_review_required",
        None if amount is None else amount > credit["senior_review_above_usd"],
        f"{_usd(amount)} vs senior review above {_usd(credit['senior_review_above_usd'])}",
        "Senior review required",
    )

    tier = "standard"
    if amount is not None:
        if amount > authority["committee_approval_above_usd"]:
            tier = "credit committee"
        elif amount > authority["senior_approval_above_usd"]:
            tier = "senior"
        elif amount > authority["manager_approval_above_usd"]:
            tier = "manager"
    add("approval_authority", tier, f"{_usd(amount)} → {tier} approval", "Approval authority")

    risk = customer.get("risk_score")
    add(
        "customer_risk_elevated",
        None if risk is None else risk >= ELEVATED_RISK_SCORE,
        f"risk_score {risk if risk is not None else 'unknown'} vs {ELEVATED_RISK_SCORE}",
        "Customer risk score elevated",
    )

    parties = state.get("account_parties") or []

    def status(p: dict) -> str:
        return (p.get("sanctions_status") or "clear").lower()

    # Blocked anywhere, or watch-listed as a co-owner of the customer's account: a hard stop.
    sanctioned = [
        p for p in parties
        if status(p) == "blocked" or (status(p) != "clear" and "co-owner" in (p.get("role") or ""))
    ]
    # Watch-listed only as a transaction counterparty: exposure a person should look at.
    watchlisted = [p for p in parties if p not in sanctioned and status(p) != "clear"]

    def describe(rows: list[dict]) -> str:
        return ", ".join(f"{p.get('name')} ({p.get('sanctions_status')}, {p.get('role')})" for p in rows)

    add(
        "sanctioned_party",
        bool(sanctioned),
        describe(sanctioned) or "no blocked party and no watch-listed co-owner",
        "Blocked party or watch-listed co-owner",
    )
    add(
        "watchlisted_counterparty",
        bool(watchlisted),
        describe(watchlisted) or f"{len(parties)} parties, none watch-listed",
        "Watch-listed transaction counterparty",
    )
    poor = [p for p in parties if (p.get("risk_rating") or "") in POOR_COUNTERPARTY_RATINGS]
    add(
        "poor_rated_counterparty",
        bool(poor),
        ", ".join(f"{p.get('name')} (rating {p.get('risk_rating')})" for p in poor)
        or "none rated " + "/".join(sorted(POOR_COUNTERPARTY_RATINGS)),
        "Poorly rated counterparty",
    )

    accounts = customer.get("accounts") or []
    frozen = [a.get("account_number") for a in accounts if a.get("status") == "frozen"]
    add(
        "frozen_account",
        bool(frozen),
        ("frozen: " + ", ".join(map(str, frozen))) if frozen else f"{len(accounts)} accounts, none frozen",
        "Frozen account",
    )

    open_fraud = [
        t.get("ticket_number")
        for t in state.get("support_tickets") or []
        if t.get("ticket_type") in OPEN_FRAUD_TICKET_TYPES and t.get("status") in OPEN_TICKET_STATUSES
    ] + [
        a.get("alert_number")
        for a in state.get("alerts") or []
        if a.get("alert_type") in OPEN_FRAUD_ALERT_TYPES and a.get("status") in OPEN_ALERT_STATUSES
    ]
    add(
        "open_fraud_case",
        bool(open_fraud),
        ("open: " + ", ".join(map(str, open_fraud))) if open_fraud else "no open fraud tickets or alerts by status",
        "Open fraud case (by status)",
    )
    return facts


# ---------------------------------------------------------------------------
# Combiner (code over facts + Jev judgments)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Escalation policy: what the bank does when a guard escalates. Written down so the
# reasoning model applies policy instead of improvising it (eval 2026-11-11: without
# this, Claude's calls on identical escalations split approve/reject).
# ``final`` is the default outcome; ``instruction`` is what Claude must check before
# departing from it. "defer" is reported as reject (re-apply once resolved).
# ---------------------------------------------------------------------------

ESCALATION_POLICY_VERSION = "2026-11-11"
ESCALATION_POLICY: dict[str, dict[str, str]] = {
    "g_unresolved_concern": {
        "final": "defer",
        "instruction": (
            "Do not raise a limit while the customer has an unresolved dispute, fraud claim or "
            "complaint with the bank. Read the cited ticket: if the text shows it is in fact "
            "settled, approve; otherwise defer until it is resolved."
        ),
    },
    "g_open_condition": {
        "final": "defer",
        "instruction": (
            "A prior approval's condition binds further increases. Read the cited decision and "
            "the rest of the evidence: approve only if a record shows the condition was met; "
            "otherwise defer until it is verified. Indirect signs (a balance or limit that looks "
            "consistent with the condition) are not verification."
        ),
    },
    "g_unexplained_alert": {
        "final": "reject",
        "instruction": (
            "Never approve credit while an alert describes unexplained fraud, structuring, mule "
            "or account-takeover activity, whatever its status field says. Reject and route to "
            "AML / fraud review, unless the evidence documents an explanation for that activity."
        ),
    },
    "g_unsure_judgment": {
        "final": "defer",
        "instruction": (
            "The fast model was unsure about the cited item. Read it yourself, decide which "
            "guard it belongs to, and apply that guard's policy. If it is still unclear, defer."
        ),
    },
    "g_precedents_against": {
        "final": "reject",
        "instruction": (
            "Most comparable precedents were rejected. Approve only if this case differs from "
            "them on the reasons they were rejected; name those reasons."
        ),
    },
    "g_evidence_incomplete": {
        "final": "defer",
        "instruction": "Required fields are missing. Defer until they are provided; do not estimate them.",
    },
    "g_income_unverified": {
        "final": "defer",
        "instruction": "The income cap cannot be applied to unverified income. Defer until income is verified.",
    },
    "g_senior_review": {
        "final": "recommend",
        "instruction": (
            "Above the senior-review amount a senior officer decides. Recommend approve or "
            "reject on the evidence and state that senior sign-off is required."
        ),
    },
    "g_customer_risk": {
        "final": "recommend",
        "instruction": (
            "The risk score is elevated but no written rule fails. Approve only if the evidence "
            "explains the score (e.g. old, resolved events); otherwise reject."
        ),
    },
    "g_watchlisted_counterparty": {
        "final": "defer",
        "instruction": (
            "The customer transacts with a watch-listed counterparty. Defer pending compliance "
            "review of that exposure."
        ),
    },
    "g_poor_counterparty": {
        "final": "recommend",
        "instruction": (
            "A counterparty is poorly rated. Approve only if the exposure is small relative to "
            "the customer's activity; otherwise reject."
        ),
    },
}


def escalation_policy_for(fired: list[str]) -> list[dict[str, str]]:
    """The policy entries for the escalating guards that fired, in guard order."""
    return [
        {"guard": g, **ESCALATION_POLICY[g]} for g in fired if g in ESCALATION_POLICY
    ]


def band(p: float | None, yes: float, no: float) -> str | None:
    if p is None:
        return None
    return "yes" if p >= yes else "no" if p <= no else "unsure"


# (fact, value that fires, reason) - written rules that can reject.
REJECT_GUARDS = [
    ("g_sanctions", "sanctioned_party", True, "a blocked party, or a watch-listed co-owner, is on the customer's accounts"),
    ("g_frozen_account", "frozen_account", True, "the customer has a frozen account"),
    ("g_open_fraud_case", "open_fraud_case", True, "an open fraud ticket or alert must be resolved first"),
    ("g_credit_score", "credit_score_meets_minimum", False, "credit score is below the policy minimum"),
    ("g_income_cap", "total_limit_within_income_cap", False, "proposed total limit exceeds the share-of-income cap"),
    ("g_increase_cap", "increase_within_current_limit_cap", False, "increase exceeds the cap relative to the current limit"),
]
# Written rules that require a person / reasoning model rather than an automatic answer.
CODE_ESCALATE_GUARDS = [
    ("g_evidence_incomplete", "evidence_complete", False, "required evidence is missing"),
    ("g_income_unverified", "income_verified", False, "income is not verified, so the income cap cannot be relied on"),
    ("g_senior_review", "senior_review_required", True, "amount requires senior review under the Credit Limit Policy"),
    ("g_customer_risk", "customer_risk_elevated", True, "customer risk score is elevated"),
    ("g_watchlisted_counterparty", "watchlisted_counterparty", True, "the customer transacts with a watch-listed counterparty"),
    ("g_poor_counterparty", "poor_rated_counterparty", True, "a counterparty on the accounts is poorly rated"),
]
# Jev judgment families that can escalate: (guard, family, reason).
JEV_ESCALATE_GUARDS = [
    ("g_unresolved_concern", "ticket_unresolved_concern", "a support ticket shows an unresolved dispute, fraud concern or complaint"),
    ("g_open_condition", "decision_open_condition", "a prior decision set a condition with no sign it was met"),
    ("g_unexplained_alert", "alert_unexplained_risk", "an alert describes risky activity that was never explained"),
]
DECIDING_FAMILIES = {family for _, family, _ in JEV_ESCALATE_GUARDS}


def combine(
    facts: dict[str, Fact],
    judgments: list[dict[str, Any]],
    yes: float,
    no: float,
) -> dict[str, Any]:
    """
    Ordered guards -> approve / reject / escalate. Rejections (written rules) are checked
    first, then code escalations, then Jev escalations; otherwise approve.
    ``judgments`` items: {id, family, ref, p, band, ...}; precedents carry ``outcome``.
    """
    guards: list[Guard] = []

    def fact_guard(name: str, fact_name: str, fires_on: bool, reason: str, effect: str) -> None:
        f = facts.get(fact_name)
        if f is None or f.value is None:
            guards.append(Guard(name, effect, "code", "not_applicable", f"{fact_name} unknown"))
        elif f.value == fires_on:
            guards.append(Guard(name, effect, "code", "fired", reason, [f"{fact_name}: {f.detail}"]))
        else:
            guards.append(Guard(name, effect, "code", "passed", f.detail))

    for g in REJECT_GUARDS:
        fact_guard(*g, effect=REJECT)
    for g in CODE_ESCALATE_GUARDS:
        fact_guard(*g, effect=ESCALATE)

    by_family: dict[str, list[dict]] = {}
    for j in judgments:
        by_family.setdefault(j["family"], []).append(j)

    for name, family, reason in JEV_ESCALATE_GUARDS:
        items = by_family.get(family, [])
        if not items:
            guards.append(Guard(name, ESCALATE, "jev", "not_applicable", "no items to judge"))
            continue
        hits = [j for j in items if j["band"] == "yes"]
        if hits:
            guards.append(
                Guard(name, ESCALATE, "jev", "fired", reason, [f"{j['ref']}: {j['p']:.2f}" for j in hits])
            )
        else:
            guards.append(Guard(name, ESCALATE, "jev", "passed", f"{len(items)} judged, none above {yes}"))

    unsure = [j for j in judgments if j["family"] in DECIDING_FAMILIES and j["band"] == "unsure"]
    guards.append(
        Guard(
            "g_unsure_judgment",
            ESCALATE,
            "jev",
            "fired" if unsure else ("passed" if judgments else "not_applicable"),
            f"Jev is unsure (between {no} and {yes}) on a deciding judgment"
            if unsure
            else "no deciding judgment in the unsure band",
            [f"{j['ref']}: {j['p']:.2f}" for j in unsure],
        )
    )

    comparable = [j for j in by_family.get("precedent_comparable", []) if j["band"] == "yes"]
    rejected = [j for j in comparable if j.get("outcome") == "rejection"]
    if not comparable:
        guards.append(Guard("g_precedents_against", ESCALATE, "jev", "not_applicable", "no comparable precedents"))
    elif len(rejected) * 2 > len(comparable):
        guards.append(
            Guard(
                "g_precedents_against",
                ESCALATE,
                "jev",
                "fired",
                f"{len(rejected)} of {len(comparable)} comparable precedents were rejected",
                [j["ref"] for j in rejected],
            )
        )
    else:
        guards.append(
            Guard(
                "g_precedents_against",
                ESCALATE,
                "jev",
                "passed",
                f"{len(rejected)} of {len(comparable)} comparable precedents were rejected",
            )
        )

    fired = [g for g in guards if g.status == "fired"]
    decider = next((g for g in fired if g.effect == REJECT), None) or next(iter(fired), None)
    outcome = decider.effect if decider else APPROVE
    return {
        "outcome": outcome,
        "routed_decision": ROUTED[outcome],
        "decided_by": (
            {"guard": decider.name, "source": decider.source, "reason": decider.reason}
            if decider
            else {"guard": None, "source": "code", "reason": "every guard passed"}
        ),
        "fired": [g.name for g in fired],
        "guards": [asdict(g) for g in guards],
    }


def facts_as_list(facts: dict[str, Fact]) -> list[dict[str, Any]]:
    return [asdict(f) for f in facts.values()]
