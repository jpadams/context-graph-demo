"""Unit tests for the Jev fast-answer pipeline (no network, no Neo4j)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app import evidence
from app.config import config
from app.jev_client import DECISION_OPTIONS, JevClient, build_decision_questions

# ---------- deterministic message parsing ----------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("She's requesting a $25,000 limit increase.", 25_000.0),
        ("approve a $25k bump", 25_000.0),
        ("a $1.5M wire", 1_500_000.0),
        ("no amount here", None),
    ],
)
def test_extract_amount(message, expected):
    assert evidence.extract_amount(message) == expected


def test_candidate_names_finds_jessica_norris():
    msg = "Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase."
    assert "Jessica Norris" in evidence.candidate_names(msg)


def test_candidate_names_finds_katherine_miller():
    msg = "We need to override the trading limit for Katherine Miller. Find precedents for similar exceptions."
    assert "Katherine Miller" in evidence.candidate_names(msg)


def test_resolve_customers_queries_graph_with_lowercased_names():
    msg = "Should we approve a credit limit increase for Jessica Norris?"
    with patch.object(evidence.context_graph_client, "execute_cypher") as cypher:
        cypher.return_value = [{"id": "p1", "name": "Jessica Norris", "risk_score": 0.2}]
        rows = evidence.resolve_customers(msg)
    assert rows[0]["id"] == "p1"
    assert "jessica norris" in cypher.call_args.args[1]["names"]


# ---------- evidence collection ----------


def test_collect_evidence_runs_every_retrieval_and_survives_failures():
    with (
        patch.object(
            evidence.context_graph_client,
            "get_customer",
            return_value={"id": "p1", "name": "J", "fast_rp_embedding": [1, 2]},
        ),
        patch.object(evidence.context_graph_client, "execute_cypher", return_value=[]),
        patch.object(
            evidence.context_graph_client, "get_customer_decisions", return_value=[{"id": "d1"}]
        ),
        patch.object(
            evidence.vector_client,
            "find_precedents_hybrid",
            side_effect=RuntimeError("no embeddings"),
        ),
        patch.object(
            evidence.context_graph_client,
            "get_policies",
            return_value=[{"name": "Credit Limit Policy"}],
        ),
    ):
        collected = evidence.collect_evidence("p1", "credit increase", "credit")

    names = [r["name"] for r in collected["retrievals"]]
    assert names == [n for n, _ in evidence.EVIDENCE_RETRIEVALS]
    assert "fast_rp_embedding" not in collected["evidence"]["customer"]
    precedents = next(r for r in collected["retrievals"] if r["name"] == "precedents")
    assert precedents["status"].startswith("failed")
    assert collected["evidence"]["precedents"] == []
    assert collected["evidence"]["policies"][0]["name"] == "Credit Limit Policy"


def test_build_decision_state_has_named_fields():
    state = evidence.build_decision_state({"summary": "x"}, {"customer": {"id": "p1"}})
    assert set(state) == {
        "request",
        "customer",
        "account_parties",
        "notable_transactions",
        "transaction_summary",
        "support_tickets",
        "alerts",
        "decision_history",
        "precedents",
        "category_outcomes",
        "policies",
    }


# ---------- Jev verdict + confidence routing ----------


def _fake_response(choice, probabilities, confidence, nouls=None):
    decision = SimpleNamespace(choice=choice, probabilities=probabilities, confidence=confidence)
    nouls = {
        k: SimpleNamespace(noul=v) for k, v in (nouls or {"elevated_customer_risk": 0.1}).items()
    }
    return SimpleNamespace(
        model="jev-test",
        choices={"decision": decision},
        nouls=nouls,
        usage=SimpleNamespace(input_tokens=100, output_tokens=0),
        request_id="req-1",
    )


def _client_with(response):
    client = JevClient()
    client._client = SimpleNamespace(system_one=lambda state, questions: response)
    return client


def test_decision_questions_offer_exactly_the_three_options():
    q = build_decision_questions({})
    assert tuple(q["decision"].criteria) == DECISION_OPTIONS


def test_questions_are_only_asked_about_present_evidence():
    empty = build_decision_questions({"precedents": [], "policies": [], "account_parties": []})
    assert "sanctioned_party_exposure" not in empty
    assert "open_fraud_activity" not in empty
    assert not [k for k in empty if k.startswith("precedent_") or k.startswith("policy_")]

    full = build_decision_questions(
        {
            "precedents": [
                {"decision_type": "approval", "category": "credit", "reasoning_summary": "ok"}
            ]
            * 2,
            "policies": [{"name": "Credit Limit Policy", "description": "d"}],
            "account_parties": [{"name": "Org", "sanctions_status": "watchlist"}],
            "support_tickets": [{"ticket_type": "fraud_report"}],
        }
    )
    assert {
        "precedent_0_supports_approval",
        "precedent_1_supports_approval",
        "policy_0_violated",
        "sanctioned_party_exposure",
        "open_fraud_activity",
    } <= set(full)


def test_confident_yes_is_kept():
    client = _client_with(
        _fake_response("yes", {"yes": 0.9, "no": 0.05, "needs_llm_decision": 0.05}, 0.9)
    )
    with patch.object(config.typesafe, "confidence_floor", 0.6):
        verdict = client.evaluate_decision({"request": {}})
    assert verdict["decision"] == "yes"
    assert verdict["routed_decision"] == "yes"
    assert verdict["probabilities"] == {"yes": 0.9, "no": 0.05, "needs_llm_decision": 0.05}
    # no precedents / policies / parties / tickets in state -> None, never a hollow number
    assert verdict["signals"]["precedents_favor_approval"] is None
    assert verdict["signals"]["violates_policy"] is None
    assert verdict["signals"]["sanctioned_party_exposure"] is None
    assert verdict["signals"]["open_fraud_activity"] is None
    assert verdict["signals"]["elevated_customer_risk"] == 0.1


def test_per_item_judgments_are_aggregated_in_code():
    state = {
        "request": {},
        "precedents": [
            {
                "id": "d1",
                "decision_type": "approval",
                "category": "credit",
                "reasoning_summary": "a",
            },
            {
                "id": "d2",
                "decision_type": "rejection",
                "category": "credit",
                "reasoning_summary": "b",
            },
        ],
        "policies": [{"id": "p1", "name": "Credit Limit Policy", "description": "x"}],
        "account_parties": [{"name": "Spencer", "sanctions_status": "watchlist"}],
    }
    nouls = {
        "precedent_0_supports_approval": 0.9,
        "precedent_1_supports_approval": 0.1,
        "policy_0_violated": 0.3,
        "sanctioned_party_exposure": 0.95,
        "elevated_customer_risk": 0.4,
        "requires_manual_review": 0.5,
        "evidence_sufficient": 0.6,
    }
    client = _client_with(
        _fake_response("no", {"yes": 0.1, "no": 0.8, "needs_llm_decision": 0.1}, 0.8, nouls)
    )
    with patch.object(config.typesafe, "confidence_floor", 0.6):
        verdict = client.evaluate_decision(state)
    assert verdict["routed_decision"] == "no"
    assert verdict["signals"]["precedents_favor_approval"] == 0.5
    assert verdict["signals"]["violates_policy"] == 0.3
    assert verdict["signals"]["sanctioned_party_exposure"] == 0.95
    assert verdict["signals"]["open_fraud_activity"] is None
    assert [p["id"] for p in verdict["item_judgments"]["precedents"]] == ["d1", "d2"]
    assert verdict["item_judgments"]["precedents"][1]["supports_approval"] == 0.1
    assert verdict["item_judgments"]["policies"][0]["violated"] == 0.3


def test_low_confidence_yes_routes_to_human():
    client = _client_with(
        _fake_response("yes", {"yes": 0.45, "no": 0.4, "needs_llm_decision": 0.15}, 0.45)
    )
    with patch.object(config.typesafe, "confidence_floor", 0.6):
        verdict = client.evaluate_decision({"request": {}})
    assert verdict["decision"] == "yes"
    assert verdict["routed_decision"] == "needs_llm_decision"
    assert "below" in verdict["routing"]["reason"]


# ---------- pipeline gating ----------


def _configured():
    return patch.object(
        type(evidence.jev_client), "is_configured", new_callable=lambda: property(lambda self: True)
    )


def test_prepare_fast_answer_skips_non_approval_messages():
    with (
        _configured(),
        patch.object(
            evidence.jev_client,
            "classify_request",
            return_value={
                "is_approval_request": 0.1,
                "category": "other",
                "request_type": "not_a_request",
            },
        ),
    ):
        result = evidence.prepare_fast_answer("Search for customer John Walsh")
    assert result["applicable"] is False
    assert result["reason"] == "not_an_approval_request"


def test_prepare_fast_answer_returns_verdict_and_graph():
    intent = {
        "is_approval_request": 0.95,
        "category": "credit",
        "request_type": "credit_limit_increase",
    }
    verdict = {"decision": "yes", "routed_decision": "yes"}
    with (
        _configured(),
        patch.object(evidence.jev_client, "classify_request", return_value=intent),
        patch.object(
            evidence, "resolve_customers", return_value=[{"id": "p1", "name": "Jessica Norris"}]
        ),
        patch.object(
            evidence,
            "collect_evidence",
            return_value={
                "evidence": {"customer": {"id": "p1"}},
                "retrievals": [{"name": "customer"}],
            },
        ),
        patch.object(evidence.jev_client, "evaluate_decision", return_value=verdict) as evaluate,
        patch.object(
            evidence, "customer_graph_data", return_value={"nodes": [], "relationships": []}
        ),
    ):
        result = evidence.prepare_fast_answer(
            "Should we approve a credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase."
        )
    assert result["applicable"] is True
    assert result["verdict"] == verdict
    assert result["request"]["requested_amount_usd"] == 25_000.0
    assert result["request"]["category"] == "credit"
    state = evaluate.call_args.args[0]
    assert state["request"]["customer_name"] == "Jessica Norris"


# ---------- final recommendation parsing ----------

from app.agent import parse_final_recommendation


def _fa(routed):
    return {"applicable": True, "verdict": {"routed_decision": routed}}


def test_escalated_case_takes_claude_final_call():
    text = "Lots of reasoning...\n\nFINAL_RECOMMENDATION: approve (confidence 0.8)"
    rec = parse_final_recommendation(text, _fa("needs_llm_decision"))
    assert rec["final"] == "approve"
    assert rec["confidence"] == 0.8
    assert rec["source"] == "claude"
    assert rec["escalated"] is True
    assert rec["agrees_with_jev"] is None


def test_jev_decided_case_records_agreement():
    rec = parse_final_recommendation("**FINAL_RECOMMENDATION: reject**", _fa("no"))
    assert rec["final"] == "reject"
    assert rec["source"] == "jev"
    assert rec["agrees_with_jev"] is True
    rec = parse_final_recommendation("FINAL_RECOMMENDATION: approve", _fa("no"))
    assert rec["agrees_with_jev"] is False


def test_missing_line_falls_back_to_jev_when_jev_decided():
    rec = parse_final_recommendation("no marker here", _fa("yes"))
    assert rec["final"] == "approve"
    assert rec["decision"] is None
    rec = parse_final_recommendation("no marker here", _fa("needs_llm_decision"))
    assert rec["final"] is None


def test_no_fast_answer_means_no_recommendation():
    assert parse_final_recommendation("FINAL_RECOMMENDATION: approve", None) is None


# ---------- parameterized case ----------

from app import cases


def test_credit_line_increase_request_is_parameterized():
    req = cases.build_credit_line_increase_request({"id": "p1", "name": "Katherine Miller"}, 40000)
    assert req["type"] == "credit_limit_increase"
    assert req["category"] == "credit"
    assert req["requested_amount_usd"] == 40000.0
    assert "Katherine Miller" in req["summary"] and "$40,000" in req["summary"]


def test_resolve_customer_by_id_name_or_search():
    with patch.object(cases.context_graph_client, "execute_cypher") as cypher:
        cypher.side_effect = [[{"id": "abc", "name": "X", "risk_score": 0.1}]]
        assert cases.resolve_customer("Jessica Norris")["id"] == "abc"
        assert cypher.call_args.args[1]["name"] == "Jessica Norris"
    with (
        patch.object(cases.context_graph_client, "execute_cypher", return_value=[]),
        patch.object(
            cases.context_graph_client,
            "search_customers",
            return_value=[{"id": "s1", "name": "Kath", "risk_score": 0.2}],
        ),
    ):
        assert cases.resolve_customer("Kath")["id"] == "s1"
    with (
        patch.object(cases.context_graph_client, "execute_cypher", return_value=[]),
        patch.object(cases.context_graph_client, "search_customers", return_value=[]),
    ):
        assert cases.resolve_customer("Nobody") is None


def test_run_credit_line_increase_uses_fixed_plan():
    with (
        patch.object(
            type(cases.jev_client),
            "is_configured",
            new_callable=lambda: property(lambda self: True),
        ),
        patch.object(
            cases, "resolve_customer", return_value={"id": "p1", "name": "Katherine Miller"}
        ),
        patch.object(
            cases,
            "collect_evidence",
            return_value={
                "evidence": {"customer": {"id": "p1"}},
                "retrievals": [{"name": "customer"}],
            },
        ) as collect,
        patch.object(cases.jev_client, "evaluate_decision", return_value={"decision": "yes"}),
        patch.object(cases, "customer_graph_data", return_value={"nodes": [], "relationships": []}),
    ):
        result = cases.run_credit_line_increase("Katherine Miller", 40000)
    assert result["applicable"] and result["case"] == "credit_limit_increase"
    assert result["request"]["requested_amount_usd"] == 40000.0
    assert collect.call_args.args[0] == "p1" and collect.call_args.args[2] == "credit"


# ---------- derived credit figures ----------


def test_derived_credit_figures_are_computed_in_code():
    request = {"requested_amount_usd": 25000.0}
    customer = {"current_credit_limit": 20000, "annual_income": 90000}
    d = evidence.derive_credit_figures(request, customer)
    assert d["proposed_total_limit_usd"] == 45000
    assert d["increase_pct_of_current_limit"] == 1.25
    assert d["proposed_total_limit_pct_of_income"] == 0.5
    assert evidence.derive_credit_figures(request, None) == {}
    assert evidence.derive_credit_figures({}, customer) == {}
    state = evidence.build_decision_state(request, {"customer": customer})
    assert state["request"]["derived"]["proposed_total_limit_usd"] == 45000
