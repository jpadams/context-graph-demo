"""
Smoke test for the Jev fast path.

    python scripts/jev_smoke.py                 # full path: Neo4j evidence + Jev verdict
    python scripts/jev_smoke.py --synthetic     # Jev only, with canned evidence (no Neo4j needed)
    python scripts/jev_smoke.py "custom question about Some Customer"

Needs TYPESAFE_API_KEY (and, without --synthetic, the Neo4j + OpenAI settings) in ../.env.
"""

import json
import sys

from app.evidence import build_decision_state, prepare_fast_answer
from app.jev_client import jev_client

JESSICA = (
    "Should we approve a credit limit increase for Jessica Norris? "
    "She's requesting a $25,000 limit increase."
)

SYNTHETIC_EVIDENCE = {
    "customer": {
        "id": "p-demo",
        "name": "Jessica Norris",
        "risk_score": 0.21,
        "accounts": [
            {"account_type": "checking", "balance": 18400.0, "status": "active"},
            {"account_type": "savings", "balance": 52000.0, "status": "active"},
        ],
    },
    "flagged_transactions": [],
    "decision_history": [
        {
            "decision_type": "approval",
            "category": "credit",
            "status": "approved",
            "reasoning_summary": "Credit line increase approved; stable income, low utilisation.",
            "confidence_score": 0.91,
        }
    ],
    "precedents": [
        {
            "decision_type": "approval",
            "category": "credit",
            "reasoning_summary": "Approved $20k increase; strong repayment history.",
            "semantic_similarity": 0.88,
        },
        {
            "decision_type": "rejection",
            "category": "credit",
            "reasoning_summary": "Rejected increase; recent flagged transactions.",
            "semantic_similarity": 0.81,
        },
    ],
    "category_outcomes": [
        {"decision_type": "approval", "status": "approved", "count": 41},
        {"decision_type": "rejection", "status": "rejected", "count": 12},
        {"decision_type": "escalation", "status": "escalated", "count": 7},
    ],
    "policies": [
        {
            "name": "Credit Limit Policy",
            "category": "credit",
            "description": "Standard credit limits based on income and credit score. Increases above $50,000 require senior review.",
        }
    ],
}


def main() -> None:
    if not jev_client.is_configured:
        sys.exit("TYPESAFE_API_KEY is not set (put it in ../.env).")

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    message = args[0] if args else JESSICA

    if "--synthetic" in sys.argv:
        print("Intent:", json.dumps(jev_client.classify_request(message), indent=2))
        request = {
            "type": "credit_limit_increase",
            "category": "credit",
            "summary": message,
            "customer_name": "Jessica Norris",
            "requested_amount_usd": 25000.0,
        }
        verdict = jev_client.evaluate_decision(build_decision_state(request, SYNTHETIC_EVIDENCE))
        print("Verdict:", json.dumps(verdict, indent=2))
        return

    result = prepare_fast_answer(message)
    if not result or not result.get("applicable"):
        print("Fast path not applicable:", json.dumps(result, indent=2, default=str))
        return
    public = {k: v for k, v in result.items() if k not in ("evidence", "graph_data")}
    print(json.dumps(public, indent=2, default=str))
    print(
        "\nEvidence sizes:",
        {k: len(v) if isinstance(v, list) else 1 for k, v in result["evidence"].items()},
    )


if __name__ == "__main__":
    main()
