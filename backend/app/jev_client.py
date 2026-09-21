"""
TypeSafe AI (Jev) integration.

Jev is a "System One" model: it evaluates a structured state and returns typed
answers with calibrated probabilities in a single pass, without generating text.

Two uses here:

1. ``classify_request`` - a fast read of the user's message: is this an
   approve/deny request, and for which category? Code uses it to decide whether
   to run the fast-answer pipeline.
2. ``evaluate_decision`` - the headline verdict over evidence collected from the
   graph: ``yes`` / ``no`` / ``needs_llm_decision`` (escalate to Claude), plus
   per-item judgments (each precedent, each policy) and reusable yes/no signals.
   Questions are only asked about evidence that is actually present; empty
   evidence yields a ``None`` signal, never a hollow probability.

Claude never sits between the graph and Jev. Code collects the evidence, Jev
decides, and Claude explains (or, when escalated, decides) afterwards with the
same evidence.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from typesafe_sdk import Choice, Noul, TypeSafeClient, TypeSafeError

from .config import config

logger = logging.getLogger(__name__)

DECISION_OPTIONS = ("yes", "no", "needs_llm_decision")
ESCALATE = "needs_llm_decision"
MAX_ITEMS = 8  # per-item questions per evidence list

REQUEST_CATEGORIES = {
    "credit": "Credit limits, loans, credit line increases",
    "trading": "Trading limits, margin, position size overrides",
    "fraud": "Suspicious activity, flagged transactions, fraud investigation",
    "compliance": "KYC, AML, regulatory verification, wire transfer checks",
    "account": "Account freezes, unfreezes, closures, access changes",
    "other": "None of the above",
}

REQUEST_TYPES = {
    "credit_limit_increase": "Raise a customer's credit limit or credit line",
    "trading_limit_override": "Exceed or override a trading or margin limit",
    "wire_transfer": "Send or approve a wire transfer",
    "account_action": "Freeze, unfreeze, close or reopen an account",
    "policy_exception": "Any other exception to a policy",
    "not_a_request": "The message is not asking for an approval or exception",
}


def build_intent_questions() -> dict[str, Choice | Noul]:
    """Questions that decide whether the fast-answer pipeline applies."""
    return {
        "is_approval_request": Noul(
            instructions=(
                "`message` asks whether something should be approved, allowed, granted, "
                "overridden or rejected for a specific customer, e.g. a credit limit "
                "increase, a trading limit override, or an exception request."
            ),
            criteria={
                "true": "The user wants an approve / deny style decision.",
                "false": "The user is asking for information, analysis, a search, or an explanation.",
            },
        ),
        "category": Choice(
            instructions="Which policy category does `message` fall under?",
            criteria=REQUEST_CATEGORIES,
        ),
        "request_type": Choice(
            instructions="What kind of request is `message` making?",
            criteria=REQUEST_TYPES,
        ),
    }


def _short(text: Any, n: int = 140) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "..."


def build_decision_questions(state: dict[str, Any]) -> dict[str, Choice | Noul]:
    """
    The judgments asked of Jev over the collected evidence, all in one call.

    ``decision`` is the headline verdict. Per-item Nouls (one per precedent, one
    per policy) make the verdict inspectable. Aggregate Nouls are only asked when
    the evidence they refer to is non-empty.
    """
    questions: dict[str, Choice | Noul] = {
        "decision": Choice(
            instructions=(
                "You are the credit and risk committee of a financial institution. Decide "
                "whether the request in `request` should be approved, using `customer` "
                "(risk_score, credit_score, annual_income, income_verified, employment_status, "
                "current_credit_limit, credit_utilization, accounts), `request.derived` "
                "(proposed total limit and its share of income, computed by code), "
                "`account_parties` (organizations that "
                "co-own the customer's accounts or transact with them, with sanctions status), "
                "`notable_transactions` and `transaction_summary` (activity on the customer's "
                "accounts), `support_tickets` and `alerts` (open issues, fraud reports), "
                "`decision_history` (past decisions about this customer), `precedents` (how "
                "similar cases were decided), `category_outcomes` (approval vs rejection base "
                "rates) and `policies` (thresholds and requirements). Missing information "
                "counts against a confident yes."
            ),
            criteria={
                "yes": {
                    "what": (
                        "Approve. The request is within policy, the customer's risk profile "
                        "is acceptable, no open fraud or sanctions concerns, and precedents "
                        "for similar cases were approved."
                    ),
                    "not_for": "Cases where a policy threshold is exceeded or risk is elevated.",
                },
                "no": {
                    "what": (
                        "Reject. The request clearly breaches a policy, or the customer shows "
                        "elevated risk (high risk_score, flagged transactions, sanctioned "
                        "parties on the accounts, open fraud reports, prior rejections), or "
                        "precedents for similar cases were rejected."
                    ),
                    "not_for": "Borderline cases where reasonable reviewers would disagree.",
                },
                "needs_llm_decision": {
                    "what": (
                        "Escalate to a reasoning model for a final approve/reject call. Evidence "
                        "is mixed, incomplete or contradictory, the request requires weighing an "
                        "exception to policy, or a policy requires case-by-case review."
                    ),
                    "not_for": "Clear-cut approvals or rejections.",
                },
            },
        ),
        "elevated_customer_risk": Noul(
            instructions=(
                "`customer` shows elevated risk: a high risk_score, entries in "
                "`notable_transactions`, frozen accounts, or prior fraud or rejection "
                "decisions in `decision_history`."
            )
        ),
        "requires_manual_review": Noul(
            instructions=(
                "A policy in `policies` requires manual or senior review for a request of "
                "this type or amount."
            )
        ),
        "evidence_sufficient": Noul(
            instructions=(
                "The evidence provided is sufficient to decide `request` without collecting "
                "more information: `customer` includes credit_score, annual_income and "
                "current_credit_limit, and the policies state the applicable thresholds."
            )
        ),
    }

    if state.get("account_parties"):
        questions["sanctioned_party_exposure"] = Noul(
            instructions=(
                "An organization in `account_parties` has a sanctions_status other than "
                "'clear', or a poor risk_rating, creating sanctions or counterparty exposure "
                "for `request`."
            )
        )
    if state.get("support_tickets") or state.get("alerts"):
        questions["open_fraud_activity"] = Noul(
            instructions=(
                "`support_tickets` or `alerts` contain an unresolved fraud report, fraud alert "
                "or suspicious-activity item concerning the customer or their accounts."
            )
        )

    for i, precedent in enumerate(state.get("precedents", [])[:MAX_ITEMS]):
        questions[f"precedent_{i}_supports_approval"] = Noul(
            instructions=(
                f"`precedents[{i}]` (a {precedent.get('decision_type')} decision in "
                f'{precedent.get("category")}: "{_short(precedent.get("reasoning_summary"))}") '
                "is a comparable case whose outcome supports approving `request`."
            ),
            criteria={
                "true": "Comparable case and its outcome favours approval.",
                "false": "Not comparable, or its outcome favours rejection or caution.",
            },
        )
    for j, policy in enumerate(state.get("policies", [])[:MAX_ITEMS]):
        questions[f"policy_{j}_violated"] = Noul(
            instructions=(
                f"`request` (see `request.derived` for the proposed total limit, its share of "
                f"income and the increase relative to the current limit) violates or exceeds "
                f"`policies[{j}]` ({policy.get('name')}: "
                f'"{_short(policy.get("description"), 260)}").'
            )
        )
    return questions


class JevClient:
    """Thin wrapper around the TypeSafe SDK with lazy initialisation."""

    def __init__(self):
        self._client: TypeSafeClient | None = None

    @property
    def is_configured(self) -> bool:
        return bool(config.typesafe.api_key)

    @property
    def model(self) -> str:
        return config.typesafe.model

    @property
    def client(self) -> TypeSafeClient:
        if self._client is None:
            if not self.is_configured:
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set. Add it to .env to enable Jev decisions."
                )
            self._client = TypeSafeClient(
                api_key=config.typesafe.api_key,
                model=config.typesafe.model,
                timeout=config.typesafe.timeout_seconds,
            )
        return self._client

    def _system_one(self, state: Any, questions: dict) -> tuple[Any, float]:
        started = time.perf_counter()
        try:
            response = self.client.system_one(state, questions)
        except TypeSafeError as e:
            logger.error(f"Jev request failed: {e}")
            raise
        return response, round((time.perf_counter() - started) * 1000)

    def classify_request(self, message: str) -> dict[str, Any]:
        """Fast intent read of the user's message."""
        response, latency_ms = self._system_one({"message": message}, build_intent_questions())
        category = response.choices["category"]
        request_type = response.choices["request_type"]
        return {
            "model": response.model,
            "is_approval_request": round(float(response.nouls["is_approval_request"].noul), 4),
            "category": category.choice,
            "category_confidence": round(float(category.confidence), 4),
            "request_type": request_type.choice,
            "request_type_confidence": round(float(request_type.confidence), 4),
            "latency_ms": latency_ms,
            "request_id": response.request_id,
        }

    def evaluate_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Ask Jev for a typed verdict on ``state``.

        Returns the raw verdict, a ``routed_decision`` that applies a confidence
        floor (escalating weak verdicts to the LLM), per-item judgments for
        precedents and policies, and signals. A signal is ``None`` when the
        evidence it depends on was empty.
        """
        questions = build_decision_questions(state)
        response, latency_ms = self._system_one(state, questions)

        decision = response.choices["decision"]
        probabilities = {
            option: round(float(decision.probabilities.get(option, 0.0)), 4)
            for option in DECISION_OPTIONS
        }
        confidence = round(float(decision.confidence), 4)
        floor = config.typesafe.confidence_floor

        routed = decision.choice
        routing_reason = f"confidence {confidence:.2f} meets the {floor:.2f} floor"
        if decision.choice != ESCALATE and confidence < floor:
            routed = ESCALATE
            routing_reason = (
                f"confidence {confidence:.2f} is below the {floor:.2f} floor, "
                "so the case is escalated to the LLM for the final call"
            )

        nouls = {name: round(float(a.noul), 4) for name, a in response.nouls.items()}
        item_judgments = summarize_item_judgments(state, nouls)
        signals = derive_signals(state, nouls, item_judgments)

        return {
            "model": response.model,
            "decision": decision.choice,
            "probabilities": probabilities,
            "confidence": confidence,
            "routed_decision": routed,
            "routing": {"confidence_floor": floor, "reason": routing_reason},
            "signals": signals,
            "item_judgments": item_judgments,
            "question_count": len(questions),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "latency_ms": latency_ms,
            "request_id": response.request_id,
        }


def summarize_item_judgments(state: dict[str, Any], nouls: dict[str, float]) -> dict[str, list]:
    """Attach each per-item probability back to the precedent / policy it judged."""
    precedents = []
    for i, p in enumerate(state.get("precedents", [])[:MAX_ITEMS]):
        prob = nouls.get(f"precedent_{i}_supports_approval")
        if prob is None:
            continue
        precedents.append(
            {
                "id": p.get("id"),
                "decision_type": p.get("decision_type"),
                "category": p.get("category"),
                "summary": _short(p.get("reasoning_summary"), 160),
                "similarity": p.get("semantic_similarity"),
                "supports_approval": prob,
            }
        )
    policies = []
    for j, pol in enumerate(state.get("policies", [])[:MAX_ITEMS]):
        prob = nouls.get(f"policy_{j}_violated")
        if prob is None:
            continue
        policies.append(
            {
                "id": pol.get("id"),
                "name": pol.get("name"),
                "description": _short(pol.get("description"), 160),
                "violated": prob,
            }
        )
    return {"precedents": precedents, "policies": policies}


def derive_signals(
    state: dict[str, Any], nouls: dict[str, float], items: dict[str, list]
) -> dict[str, float | None]:
    """
    Reusable signals. Aggregates are computed in code from per-item judgments;
    a signal is None when the underlying evidence list was empty.
    """
    precedents = items["precedents"]
    policies = items["policies"]
    return {
        "precedents_favor_approval": (
            round(sum(p["supports_approval"] for p in precedents) / len(precedents), 4)
            if precedents
            else None
        ),
        "violates_policy": (round(max(p["violated"] for p in policies), 4) if policies else None),
        "sanctioned_party_exposure": nouls.get("sanctioned_party_exposure")
        if state.get("account_parties")
        else None,
        "open_fraud_activity": nouls.get("open_fraud_activity")
        if (state.get("support_tickets") or state.get("alerts"))
        else None,
        "elevated_customer_risk": nouls.get("elevated_customer_risk"),
        "requires_manual_review": nouls.get("requires_manual_review"),
        "evidence_sufficient": nouls.get("evidence_sufficient"),
    }


# Global client instance
jev_client = JevClient()
