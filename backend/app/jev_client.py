"""
TypeSafe AI (Jev) integration.

Jev is a "System One" model: it evaluates a structured state and returns typed
answers with calibrated probabilities in a single pass, without generating text.

Three uses here:

1. ``classify_request`` - a fast read of the user's message: is this an
   approve/deny request, and for which category? Code uses it to decide whether
   to run the fast-answer pipeline.
2. ``evaluate_judgments`` (default, "decomposed") - Jev makes *judgments*, code
   makes the *decision*. One request asks a narrow Noul per evidence item that
   needs reading, not computing: is this ticket an unresolved concern, did this
   prior decision leave a condition unmet, is this alert unexplained, is this
   precedent comparable? ``decision_policy.combine`` turns the answers plus the
   code-computed facts into approve / reject / escalate with a guard trace.
3. ``evaluate_decision`` ("monolithic", kept as the eval baseline) - one Choice
   with business rules in its criteria returns ``yes`` / ``no`` /
   ``needs_llm_decision``, plus supporting Nouls.

Claude never sits between the graph and Jev. Code collects the evidence, Jev
judges, code decides, and Claude explains (or, when escalated, decides)
afterwards with the same evidence.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from typesafe_sdk import Choice, Noul, TypeSafeClient, TypeSafeError

from .config import config
from .decision_policy import band

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


# ---------------------------------------------------------------------------
# Decomposed judgments: one narrow question per evidence item that needs reading.
# Families map to guards in decision_policy.JEV_ESCALATE_GUARDS.
# ---------------------------------------------------------------------------

JUDGMENT_FAMILIES = {
    "ticket_unresolved_concern": "support_tickets",
    "decision_open_condition": "decision_history",
    "alert_unexplained_risk": "alerts",
    "precedent_comparable": "precedents",
}


def _ticket_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f"Judging from what `support_tickets[{i}].subject` and `support_tickets[{i}].description` "
            "say, the customer still has an unresolved dispute, fraud concern or complaint with "
            f"the bank. Read the text itself: `support_tickets[{i}].status` may say resolved or "
            "closed even when the customer's own words say otherwise."
        ),
        criteria={
            "true": (
                "The text shows the customer still disputing a charge, still waiting on a refund "
                "or fraud investigation, or still unhappy and escalating a complaint."
            ),
            "false": (
                "The text shows the issue was settled to the customer's satisfaction, or the "
                "ticket is a routine request (password, statements, questions) with no grievance."
            ),
        },
    )


def _decision_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f"`decision_history[{i}].reasoning` attaches a condition, requirement or deadline to "
            "its outcome (for example re-verifying income or fixing an overdrawn account), and no "
            "other entry in `decision_history` or `support_tickets` states that this condition "
            "was met."
        ),
        criteria={
            "true": "A condition was attached and no later record says it was satisfied.",
            "false": "No condition was attached, or a later record states it was satisfied.",
        },
    )


def _alert_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f"Judging from what `alerts[{i}].description` says, the alert describes suspicious "
            "activity (possible fraud, money laundering, structuring, account takeover or "
            "sanctions exposure) that has not been explained or cleared by an analyst. Read the "
            f"text itself: `alerts[{i}].status` may say resolved or false_positive even when no "
            "explanation was recorded."
        ),
        criteria={
            "true": "Suspicious activity is described and no documented explanation clears it.",
            "false": "The text documents a benign explanation or an analyst verification that clears it.",
        },
    )


def _precedent_question(i: int) -> Noul:
    return Noul(
        instructions=(
            f"`precedents[{i}]` is a comparable case for deciding `request`: it concerns the same "
            "kind of request for a customer in a broadly similar situation, so its outcome tells "
            "us something about how `request` should be decided."
        ),
        criteria={
            "true": "Same kind of request and a similar situation.",
            "false": "A different kind of request (fraud, trading, compliance...) or an unrelated situation.",
        },
    )


_QUESTION_BUILDERS = {
    "ticket_unresolved_concern": _ticket_question,
    "decision_open_condition": _decision_question,
    "alert_unexplained_risk": _alert_question,
    "precedent_comparable": _precedent_question,
}


def _item_ref(family: str, item: dict, i: int) -> tuple[str, str]:
    """(stable item id for eval joins, short human label) for an evidence item."""
    if family == "ticket_unresolved_concern":
        return str(item.get("ticket_number")), f"{item.get('ticket_number')} ({item.get('ticket_type')}, {item.get('status')})"
    if family == "alert_unexplained_risk":
        return str(item.get("alert_number")), f"{item.get('alert_number')} ({item.get('alert_type')}, {item.get('status')})"
    if family == "decision_open_condition":
        return str(item.get("id")), f"decision {str(item.get('id'))[:8]} ({item.get('category')} {item.get('decision_type')})"
    return str(item.get("id")), f"precedent {str(item.get('id'))[:8]} ({item.get('category')} {item.get('decision_type')})"


def _judgeable(family: str, item: dict) -> bool:
    """Only ask about items that carry the text the question reads."""
    if family == "ticket_unresolved_concern":
        return bool(item.get("description") or item.get("subject"))
    if family == "alert_unexplained_risk":
        return bool(item.get("description"))
    if family == "decision_open_condition":
        return bool(item.get("reasoning"))
    return True


def build_judgment_questions(state: dict[str, Any]) -> tuple[dict[str, Noul], dict[str, dict]]:
    """
    One Noul per judgeable evidence item, all independent, sent in one request.
    Returns (questions, index) where index maps question id -> item metadata.
    Empty evidence asks nothing.
    """
    questions: dict[str, Noul] = {}
    index: dict[str, dict] = {}
    for family, field in JUDGMENT_FAMILIES.items():
        for i, item in enumerate((state.get(field) or [])[:MAX_ITEMS]):
            if not isinstance(item, dict) or not _judgeable(family, item):
                continue
            qid = f"{family}_{i}"
            questions[qid] = _QUESTION_BUILDERS[family](i)
            item_id, ref = _item_ref(family, item, i)
            index[qid] = {
                "id": qid,
                "family": family,
                "path": f"{field}[{i}]",
                "item_id": item_id,
                "ref": ref,
                **({"outcome": item.get("decision_type")} if family == "precedent_comparable" else {}),
            }
    return questions, index


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

    def evaluate_judgments(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Ask Jev the per-item judgment questions for ``state`` in one request and band
        each answer (yes / unsure / no). No decision here: see ``decision_policy``.
        """
        questions, index = build_judgment_questions(state)
        yes, no = config.typesafe.yes_threshold, config.typesafe.no_threshold
        if not questions:
            return {
                "model": self.model,
                "judgments": [],
                "question_count": 0,
                "latency_ms": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "request_id": None,
                "thresholds": {"yes": yes, "no": no},
            }
        response, latency_ms = self._system_one(state, questions)
        judgments = []
        for qid, meta in index.items():
            answer = response.nouls.get(qid)
            if answer is None:
                # Fail closed: a question Jev did not answer counts as unsure (escalates),
                # never as a pass.
                judgments.append({**meta, "p": 0.5, "band": "unsure", "missing": True})
                continue
            p = round(float(answer.noul), 4)
            judgments.append({**meta, "p": p, "band": band(p, yes, no)})
        return {
            "model": response.model,
            "judgments": judgments,
            "question_count": len(questions),
            "latency_ms": latency_ms,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "request_id": response.request_id,
            "thresholds": {"yes": yes, "no": no},
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
            "mode": "monolithic",
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
