"""
Narrative text for the sample graph: support tickets, alerts, transaction memos and
conditional credit decisions.

The original generator filled these fields with Faker lorem ipsum, which gives a semantic
model nothing to judge. These templates read like real case notes, and some of them
deliberately contradict the structured field next to them (a ticket whose ``status`` is
``resolved`` while the customer's reply says it is not). Code reading ``status`` sees one
thing; a model reading the text sees another. Those contradictions are what the
decomposed Jev judgments are for.

Every rewritten item gets a ground-truth label, written to ``evals/narrative_labels.json``
(never to the graph), so per-item Jev judgments can be scored.

All choices use a ``random.Random`` seeded from the node id, so rewriting is deterministic
and does not disturb the generator's global random stream.
"""

from __future__ import annotations

import random

CONCERN_TICKET_TYPES = {"transaction_dispute", "fraud_report", "complaint"}
OPEN_TICKET_STATUSES = {"open", "in_progress", "pending_customer"}
OPEN_ALERT_STATUSES = {"open", "investigating", "escalated"}
CONTRADICTION_RATE = 0.3

MERCHANTS = [
    "QuickMart #2291",
    "Harbor Electronics",
    "SkyLine Travel",
    "Delta Fuel 114",
    "StreamBox Media",
    "Northgate Pharmacy",
    "Brightway Furniture",
    "CityRide",
]
PAYEES = ["J. Alvarez", "M. Chen", "Peak Property Mgmt", "R. Okafor", "Lakeside Dental"]
CITIES = ["Lisbon", "Toronto", "Mexico City", "Denver", "Dublin", "Singapore"]


def _money(rng: random.Random, lo: int, hi: int) -> str:
    return f"${rng.randint(lo, hi):,}"


# ---------------------------------------------------------------------------
# Support tickets. Label: unresolved_concern - the customer still has an open
# dispute, fraud concern or complaint, judged from the text, whatever `status` says.
# ---------------------------------------------------------------------------


def _concern_ongoing(ticket_type: str, rng: random.Random) -> tuple[str, str]:
    m, amt = rng.choice(MERCHANTS), _money(rng, 120, 4800)
    if ticket_type == "fraud_report":
        return (
            f"Unrecognised card charges at {m}",
            f"Customer reports {rng.randint(2, 5)} card transactions at {m} totalling {amt} that "
            "they did not make. Card was in their possession. Card blocked and replacement "
            "ordered; investigation of the charges is ongoing and provisional credit has not "
            "yet been issued.",
        )
    if ticket_type == "complaint":
        return (
            "Complaint about repeated overdraft fees",
            f"Customer is unhappy about {rng.randint(3, 6)} overdraft fees charged in one week "
            "while a deposit was on hold. Asked for a supervisor and says they will move their "
            "accounts if the fees are not reversed. Awaiting decision from branch manager.",
        )
    return (
        f"Disputed charge {amt} from {m}",
        f"Customer disputes a {amt} charge from {m}: ordered goods were never delivered and the "
        "merchant is not responding. Chargeback filed; waiting on merchant response.",
    )


def _concern_resolved(ticket_type: str, rng: random.Random) -> tuple[str, str]:
    m, amt = rng.choice(MERCHANTS), _money(rng, 120, 4800)
    if ticket_type == "fraud_report":
        return (
            f"Unrecognised card charges at {m}",
            f"Customer reported charges at {m} totalling {amt}. Investigation confirmed card-"
            "not-present fraud; all charges permanently credited back and a new card issued. "
            "Customer confirmed by phone they are satisfied and the matter is closed.",
        )
    if ticket_type == "complaint":
        return (
            "Complaint about overdraft fees",
            "Customer complained about overdraft fees during a deposit hold. Fees reversed as a "
            "courtesy and low-balance alerts set up. Customer thanked the agent; no further "
            "action needed.",
        )
    return (
        f"Disputed charge {amt} from {m}",
        f"Customer disputed {amt} from {m}. Merchant issued a full refund, which posted to the "
        "account. Customer confirmed receipt of the refund and asked us to close the case.",
    )


def _concern_closed_but_unresolved(ticket_type: str, rng: random.Random) -> tuple[str, str]:
    """Status says resolved/closed; the text says the customer is still unhappy."""
    m, amt = rng.choice(MERCHANTS), _money(rng, 300, 4800)
    days = rng.randint(10, 21)
    if ticket_type == "fraud_report":
        return (
            f"Unrecognised card charges at {m}",
            f"Auto-closed after {days} days with no agent activity. Customer wrote back after "
            f"closure: 'I still have not been refunded the {amt} in fraudulent charges from "
            f"{m}. This is the third time I have contacted you and nobody has called me back.'",
        )
    if ticket_type == "complaint":
        return (
            "Complaint about overdraft fees",
            "Marked resolved after a standard apology letter was sent. Customer replied that "
            "the fees were never refunded, called the response 'insulting', and says they have "
            "filed a complaint with the consumer finance regulator.",
        )
    return (
        f"Disputed charge {amt} from {m}",
        f"Closed after the merchant submitted a delivery receipt. Customer replied the next "
        f"day that the signature on the receipt is not theirs, the {amt} charge is still on "
        "their statement, and they want the dispute reopened.",
    )


def _routine(ticket_type: str, status: str, rng: random.Random) -> tuple[str, str]:
    done = status not in OPEN_TICKET_STATUSES
    if ticket_type == "password_reset":
        return (
            "Locked out of online banking",
            "Customer locked out after too many password attempts. Identity verified and reset "
            "link sent." + (" Customer logged in successfully." if done else ""),
        )
    if ticket_type == "statement_request":
        return (
            "Request for past statements",
            f"Customer requested {rng.randint(3, 12)} months of statements for a mortgage "
            "application." + (" PDFs sent to secure inbox." if done else ""),
        )
    if ticket_type == "limit_increase":
        return (
            "Asking about a higher credit limit",
            "Customer asked how to request a higher credit limit and what documents are needed. "
            "Explained the process and income verification requirements.",
        )
    if ticket_type == "account_inquiry":
        return (
            "Question about savings interest rate",
            "Customer asked why the savings rate changed this quarter. Explained the variable "
            "rate terms" + (" and customer had no further questions." if done else "."),
        )
    return (
        "General question about mobile app",
        "Customer asked how to set up mobile cheque deposit. Walked through the steps"
        + (" and the first deposit went through." if done else "."),
    )


def ticket_narrative(ticket_id: str, ticket_type: str, status: str) -> tuple[str, str, dict]:
    """(subject, description, label) for a support ticket."""
    rng = random.Random(f"ticket:{ticket_id}")
    if ticket_type in CONCERN_TICKET_TYPES:
        if status in OPEN_TICKET_STATUSES:
            subject, text = _concern_ongoing(ticket_type, rng)
            label = {"unresolved_concern": True, "variant": "open_concern"}
        elif rng.random() < CONTRADICTION_RATE:
            subject, text = _concern_closed_but_unresolved(ticket_type, rng)
            label = {"unresolved_concern": True, "variant": "closed_but_unresolved"}
        else:
            subject, text = _concern_resolved(ticket_type, rng)
            label = {"unresolved_concern": False, "variant": "resolved_concern"}
    else:
        subject, text = _routine(ticket_type, status, rng)
        label = {"unresolved_concern": False, "variant": "routine"}
    return subject, text, label


# ---------------------------------------------------------------------------
# Alerts. Label: unexplained_risk - the description shows suspicious activity that
# has not been explained or cleared, judged from the text, whatever `status` says.
# ---------------------------------------------------------------------------


def _alert_suspicious(alert_type: str, rng: random.Random) -> str:
    n = rng.randint(5, 9)
    if alert_type in ("pattern_match", "amount_threshold"):
        return (
            f"{n} cash deposits between $9,200 and $9,900 at {rng.randint(2, 4)} different branches "
            f"over {rng.randint(6, 12)} days, each just below the $10,000 reporting threshold. "
            "Source of funds not documented."
        )
    if alert_type == "sanctions_hit":
        return (
            "Outgoing wire beneficiary name is a partial match to an entry on the consolidated "
            "sanctions list. Beneficiary bank is in a high-risk jurisdiction; match not yet "
            "discounted."
        )
    if alert_type in ("geographic_anomaly", "device_change"):
        return (
            f"Login from a new device in {rng.choice(CITIES)} followed within {rng.randint(3, 20)} "
            f"minutes by a change of phone number and a {_money(rng, 2000, 9000)} transfer to a "
            "first-time payee. Customer could not be reached to confirm."
        )
    return (
        f"Account received {n} incoming transfers from unrelated individuals within 48 hours, "
        "each forwarded out by wire the same day. Pattern consistent with a money mule account."
    )


def _alert_benign(alert_type: str, rng: random.Random) -> str:
    city = rng.choice(CITIES)
    if alert_type in ("pattern_match", "amount_threshold"):
        return (
            f"Large deposit of {_money(rng, 12000, 40000)} flagged by amount rule. Documented as "
            "proceeds from the sale of a vehicle; bill of sale on file. Cleared by analyst."
        )
    if alert_type == "sanctions_hit":
        return (
            "Name screening hit on payee. Analyst compared date of birth and address; different "
            "person from the listed entity. Match discounted as false positive."
        )
    if alert_type in ("geographic_anomaly", "device_change"):
        return (
            f"Card used in {city}. Customer had filed a travel notice for {city} and confirmed the "
            "purchases by SMS one-time passcode. Cleared."
        )
    return (
        f"Spending spike of {_money(rng, 3000, 9000)} flagged by behavioural model. Customer "
        "confirmed by phone callback it was a planned home renovation purchase. Cleared."
    )


def alert_narrative(alert_id: str, alert_type: str, status: str) -> tuple[str, dict]:
    """(description, label) for an alert."""
    rng = random.Random(f"alert:{alert_id}")
    title = alert_type.replace("_", " ").title()
    if status in OPEN_ALERT_STATUSES:
        return (
            f"{title}: {_alert_suspicious(alert_type, rng)} Under review.",
            {"unexplained_risk": True, "variant": "open_suspicious"},
        )
    if rng.random() < CONTRADICTION_RATE:
        return (
            f"{title}: Closed as {status.replace('_', ' ')} by bulk closure rule during backlog "
            f"clean-up. Original finding: {_alert_suspicious(alert_type, rng)} No analyst notes.",
            {"unexplained_risk": True, "variant": "closed_but_unexplained"},
        )
    return (
        f"{title}: {_alert_benign(alert_type, rng)}",
        {"unexplained_risk": False, "variant": "cleared_benign"},
    )


# ---------------------------------------------------------------------------
# Transaction memos (no label; realism only).
# ---------------------------------------------------------------------------


def transaction_memo(txn_id: str, txn_type: str | None) -> str:
    rng = random.Random(f"txn:{txn_id}")
    if txn_type == "deposit":
        return rng.choice(
            [
                f"Payroll deposit {rng.choice(['ACME CORP', 'NORTHWIND LLC', 'STATE OF CO'])}",
                "Mobile cheque deposit",
                "Cash deposit at branch",
                "Tax refund IRS TREAS 310",
            ]
        )
    if txn_type == "withdrawal":
        return rng.choice(
            ["ATM withdrawal", f"Card purchase {rng.choice(MERCHANTS)}", "Cash withdrawal at branch"]
        )
    if txn_type == "trade":
        return rng.choice(
            ["Buy 50 sh VTI", "Sell 20 sh AAPL", "Dividend reinvestment", "Buy 100 sh BND"]
        )
    return rng.choice(
        [
            f"Zelle to {rng.choice(PAYEES)}",
            "Transfer to savings",
            f"Online bill pay {rng.choice(PAYEES)}",
            "Rent payment",
        ]
    )


# ---------------------------------------------------------------------------
# Conditional credit decisions. Label: open_condition - the decision attached a
# condition and nothing shows it was met.
# ---------------------------------------------------------------------------

CONDITIONS = [
    (
        "re-verification of income within 60 days (two recent pay stubs or the latest tax "
        "return)",
        "Income re-verification completed: two recent pay stubs received and match the stated "
        "income. Condition from the earlier credit approval is satisfied.",
    ),
    (
        "the customer bringing the overdrawn checking account back to a positive balance within "
        "30 days",
        "Checking account has been in positive balance for 45 consecutive days. Condition from "
        "the earlier credit approval is satisfied.",
    ),
    (
        "proof of current employment from the new employer, as the customer changed jobs last "
        "month",
        "Employment letter from the new employer received and verified by phone. Condition "
        "from the earlier credit approval is satisfied.",
    ),
]


def conditional_credit_decision(decision_id: str, new_limit: int | None = None) -> tuple[str, str]:
    """
    (reasoning, follow_up_reasoning) for a conditional credit approval. Pass ``new_limit``
    (the customer's current limit) so the approved limit matches the customer record.
    """
    rng = random.Random(f"decision:{decision_id}")
    condition, follow_up = rng.choice(CONDITIONS)
    new = new_limit or rng.randint(9, 15) * 1000
    old = int(new * rng.uniform(0.5, 0.75) / 500) * 500
    reasoning = (
        f"Credit limit increase from ${old:,} to ${new:,} approved conditionally, subject to "
        f"{condition}. If the condition is not met, the limit reverts and no further increases "
        "should be granted until it is."
    )
    return reasoning, follow_up


def summary(text: str) -> str:
    return text[:100] + "..." if len(text) > 100 else text
