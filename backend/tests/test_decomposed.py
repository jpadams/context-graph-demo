"""Decomposed decisions: code facts + per-item Jev judgments -> code guards (no network)."""

from unittest.mock import patch

from app import cases
from app.decision_policy import band, combine, compute_facts
from app.jev_client import build_judgment_questions

YES, NO = 0.7, 0.3


def _state(**overrides):
    state = {
        "request": {
            "requested_amount_usd": 5000.0,
            "derived": {
                "proposed_total_limit_usd": 20000,
                "proposed_total_limit_pct_of_income": 0.2,
                "increase_pct_of_current_limit": 0.33,
            },
        },
        "customer": {
            "credit_score": 760,
            "annual_income": 100000,
            "current_credit_limit": 15000,
            "income_verified": True,
            "risk_score": 0.2,
            "accounts": [{"account_number": "A1", "status": "active"}],
        },
        "account_parties": [],
        "support_tickets": [],
        "alerts": [],
        "decision_history": [],
        "precedents": [],
        "policies": [],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(state.get(key), dict):
            state[key] = {**state[key], **value}
        else:
            state[key] = value
    return state


def _j(family, p, ref="item", **extra):
    return {"id": f"{family}_0", "family": family, "path": "x[0]", "ref": ref, "p": p, "band": band(p, YES, NO), **extra}


def test_clean_case_approves_when_every_guard_passes():
    result = combine(compute_facts(_state()), [], YES, NO)
    assert result["outcome"] == "approve" and result["routed_decision"] == "yes"
    assert result["decided_by"]["guard"] is None


def test_policy_breach_rejects_from_code():
    state = _state(customer={"credit_score": 600})
    result = combine(compute_facts(state), [], YES, NO)
    assert result["outcome"] == "reject"
    assert result["decided_by"] == {
        "guard": "g_credit_score",
        "source": "code",
        "reason": "credit score is below the policy minimum",
    }


def test_rejection_takes_precedence_over_escalation():
    state = _state(
        customer={"income_verified": False},
        account_parties=[{"name": "X", "sanctions_status": "blocked", "role": "co-owner of account"}],
    )
    result = combine(compute_facts(state), [], YES, NO)
    assert result["outcome"] == "reject"
    assert result["decided_by"]["guard"] == "g_sanctions"
    assert "g_income_unverified" in result["fired"]


def test_jev_judgment_escalates_but_never_rejects():
    judgments = [_j("ticket_unresolved_concern", 0.96, ref="TKT1")]
    result = combine(compute_facts(_state()), judgments, YES, NO)
    assert result["outcome"] == "escalate" and result["routed_decision"] == "needs_llm_decision"
    assert result["decided_by"]["guard"] == "g_unresolved_concern"
    assert result["decided_by"]["source"] == "jev"


def test_unsure_judgment_escalates_instead_of_rounding():
    judgments = [_j("decision_open_condition", 0.5)]
    result = combine(compute_facts(_state()), judgments, YES, NO)
    assert result["decided_by"]["guard"] == "g_unsure_judgment"


def test_low_judgments_approve():
    judgments = [_j("ticket_unresolved_concern", 0.03), _j("alert_unexplained_risk", 0.05)]
    assert combine(compute_facts(_state()), judgments, YES, NO)["outcome"] == "approve"


def test_comparable_rejected_precedents_escalate_code_reads_outcome():
    judgments = [
        _j("precedent_comparable", 0.9, ref="P1", outcome="rejection"),
        _j("precedent_comparable", 0.9, ref="P2", outcome="rejection"),
        _j("precedent_comparable", 0.9, ref="P3", outcome="approval"),
        _j("precedent_comparable", 0.1, ref="P4", outcome="approval"),  # not comparable
    ]
    result = combine(compute_facts(_state()), judgments, YES, NO)
    assert result["decided_by"]["guard"] == "g_precedents_against"


def test_unknown_facts_do_not_reject_but_missing_evidence_escalates():
    state = _state(customer={"credit_score": None})
    facts = compute_facts(state)
    assert facts["credit_score_meets_minimum"].value is None
    result = combine(facts, [], YES, NO)
    guards = {g["name"]: g for g in result["guards"]}
    assert guards["g_credit_score"]["status"] == "not_applicable"
    assert result["decided_by"]["guard"] == "g_evidence_incomplete"


def test_policy_rules_come_from_the_graph():
    state = _state(
        policies=[{"name": "Credit Limit Policy", "threshold_rules": '{"min_credit_score": 780}'}]
    )
    assert compute_facts(state)["credit_score_meets_minimum"].value is False


def test_open_fraud_by_status_is_code_closed_by_text_is_jev():
    open_ticket = {"ticket_number": "T1", "ticket_type": "fraud_report", "status": "open"}
    closed_ticket = {"ticket_number": "T2", "ticket_type": "fraud_report", "status": "resolved"}
    assert compute_facts(_state(support_tickets=[open_ticket]))["open_fraud_case"].value is True
    assert compute_facts(_state(support_tickets=[closed_ticket]))["open_fraud_case"].value is False


def test_judgment_questions_one_per_item_and_only_for_present_text():
    state = _state(
        support_tickets=[
            {"ticket_number": "T1", "ticket_type": "complaint", "status": "resolved", "description": "x"},
            {"ticket_number": "T2", "ticket_type": "complaint", "status": "resolved"},  # no text
        ],
        alerts=[{"alert_number": "A1", "alert_type": "pattern_match", "status": "false_positive", "description": "y"}],
        decision_history=[{"id": "d1", "reasoning": "approved subject to ..."}],
        precedents=[{"id": "p1", "decision_type": "rejection", "category": "credit"}],
    )
    questions, index = build_judgment_questions(state)
    assert set(questions) == {
        "ticket_unresolved_concern_0",
        "alert_unexplained_risk_0",
        "decision_open_condition_0",
        "precedent_comparable_0",
    }
    assert "`support_tickets[0].description`" in questions["ticket_unresolved_concern_0"].instructions
    assert index["ticket_unresolved_concern_0"]["item_id"] == "T1"
    assert index["precedent_comparable_0"]["outcome"] == "rejection"
    assert build_judgment_questions(_state()) == ({}, {})


def test_decide_dispatches_by_mode():
    judged = {
        "model": "jev",
        "judgments": [_j("ticket_unresolved_concern", 0.95, ref="TKT1")],
        "thresholds": {"yes": YES, "no": NO},
        "question_count": 1,
        "usage": {},
        "latency_ms": 5,
        "request_id": "r",
    }
    with (
        patch.object(cases.jev_client, "evaluate_judgments", return_value=judged),
        patch.object(cases.jev_client, "evaluate_decision", return_value={"mode": "monolithic"}) as mono,
    ):
        decomposed = cases.decide(_state(), "decomposed")
        monolithic = cases.decide(_state(), "monolithic")
    assert decomposed["mode"] == "decomposed"
    assert decomposed["routed_decision"] == "needs_llm_decision"
    assert decomposed["decided_by"]["guard"] == "g_unresolved_concern"
    assert {f["name"] for f in decomposed["facts"]} >= {"credit_score_meets_minimum", "sanctioned_party"}
    assert monolithic == {"mode": "monolithic"} and mono.called


def test_watchlisted_counterparty_escalates_but_watchlisted_co_owner_rejects():
    counterparty = {"name": "W", "sanctions_status": "watchlist", "role": "transaction counterparty"}
    co_owner = {"name": "W", "sanctions_status": "watchlist", "role": "co-owner of account"}
    r1 = combine(compute_facts(_state(account_parties=[counterparty])), [], YES, NO)
    r2 = combine(compute_facts(_state(account_parties=[co_owner])), [], YES, NO)
    assert r1["decided_by"]["guard"] == "g_watchlisted_counterparty" and r1["outcome"] == "escalate"
    assert r2["decided_by"]["guard"] == "g_sanctions" and r2["outcome"] == "reject"


def test_escalated_verdict_carries_the_written_policy_for_fired_guards():
    from app.agent import format_fast_answer_context

    judged = {
        "model": "jev",
        "judgments": [_j("decision_open_condition", 0.95, ref="decision d1")],
        "thresholds": {"yes": YES, "no": NO},
        "question_count": 1,
        "usage": {},
        "latency_ms": 5,
        "request_id": "r",
    }
    with patch.object(cases.jev_client, "evaluate_judgments", return_value=judged):
        verdict = cases.decide(_state(), "decomposed")
    entries = verdict["escalation_policy"]["entries"]
    assert [e["guard"] for e in entries] == ["g_open_condition"]
    assert entries[0]["final"] == "defer"
    context = format_fast_answer_context({"verdict": verdict, "retrievals": [], "evidence": {}})
    assert "escalation policy" in context and "Indirect signs" in context


def test_approved_verdict_has_no_escalation_policy():
    judged = {
        "model": "jev",
        "judgments": [],
        "thresholds": {"yes": YES, "no": NO},
        "question_count": 0,
        "usage": {},
        "latency_ms": 0,
        "request_id": None,
    }
    with patch.object(cases.jev_client, "evaluate_judgments", return_value=judged):
        assert cases.decide(_state(), "decomposed")["escalation_policy"] is None


def test_unanswered_judgment_fails_closed():
    from types import SimpleNamespace

    from app.jev_client import JevClient

    state = _state(
        support_tickets=[{"ticket_number": "T1", "ticket_type": "complaint", "status": "resolved", "description": "x"}]
    )
    response = SimpleNamespace(
        nouls={}, model="jev", request_id="r", usage=SimpleNamespace(input_tokens=1, output_tokens=1)
    )
    client = JevClient()
    with patch.object(client, "_system_one", return_value=(response, 3)):
        judged = client.evaluate_judgments(state)
    assert judged["judgments"][0]["band"] == "unsure" and judged["judgments"][0]["missing"]
    result = combine(compute_facts(state), judged["judgments"], YES, NO)
    assert result["outcome"] == "escalate" and result["decided_by"]["guard"] == "g_unsure_judgment"
