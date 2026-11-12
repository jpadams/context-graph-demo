#!/usr/bin/env python3
"""
Calibration and stability of the per-item Jev judgments (decomposed mode).

For every customer with labeled evidence (tickets, alerts, conditional decisions in
evals/narrative_labels.json, plus the planted cases), build the same state the fast
path builds for those three evidence lists, ask the judgment questions, and score
each answer against its label. Reports:

* a reliability table: predictions binned by probability vs. the observed yes-rate,
  with the Brier score (calibration, as the build guide recommends: plot it, don't assume it);
* accuracy / unsure-rate / errors at several yes/no cut-offs (threshold tuning on labels);
* repeat-run stability: how often the band of the same question changes across runs.

Precedents are not judged here (their "comparable" label is not defined).
Labels come from templates written for this demo, so this measures in-distribution
behaviour; it is not a substitute for labeled production data.

    cd backend && set -a && source ../.env && set +a
    .venv/bin/python scripts/eval_judgments.py [--repeats 2] [--limit 60]
    .venv/bin/python scripts/eval_judgments.py --hard [--repeats 3]

--hard runs evals/hard_items.json instead: held-out items written independently of the
generator templates (partial refunds, sarcasm, negation, mixed signals). Items labeled
null are genuinely ambiguous; the right answer for them is "unsure" (escalate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import evidence  # noqa: E402
from app.context_graph_client import context_graph_client  # noqa: E402
from app.jev_client import jev_client  # noqa: E402
from eval_decision_paths import FAMILY_LABEL, load_item_labels  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"
CUTOFFS = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]


def customers_with_labeled_items() -> list[str]:
    rows = context_graph_client.execute_cypher(
        """
        MATCH (p:Person)
        WHERE EXISTS { (p)<-[:SUBMITTED_BY]-(:SupportTicket) }
           OR EXISTS { (p)-[:OWNS]->(:Account)<-[:REGARDING]-(:SupportTicket) }
           OR EXISTS { (p)-[:OWNS]->(:Account)<-[:REGARDING]-(:Alert) }
           OR EXISTS { (p)-[:OWNS]->(:Account)<-[:FROM_ACCOUNT|TO_ACCOUNT]-(:Transaction)<-[:TRIGGERED_BY]-(:Alert) }
           OR EXISTS { (:Decision {category: 'credit'})-[:ABOUT]->(p) }
        RETURN p.id AS id ORDER BY id
        """
    )
    return [r["id"] for r in rows]


def judge_customer(customer_id: str) -> list[dict]:
    state = evidence.slim(
        {
            "request": {"type": "credit_limit_increase", "category": "credit"},
            "support_tickets": evidence._support_tickets(customer_id),
            "alerts": evidence._alerts(customer_id),
            "decision_history": context_graph_client.get_customer_decisions(
                customer_id=customer_id, limit=10
            ),
        }
    )
    judged = jev_client.evaluate_judgments(state)
    return [{**j, "customer_id": customer_id, "model": judged["model"]} for j in judged["judgments"]]


def run_once(customers: list[str]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return [j for batch in pool.map(judge_customer, customers) for j in batch]


def score(judgments: list[dict], truth: dict) -> dict:
    labeled = []
    for j in judgments:
        group, _ = FAMILY_LABEL.get(j["family"], (None, None))
        if group and j["item_id"] in truth[group] and not j.get("missing"):
            labeled.append((j, truth[group][j["item_id"]]))
    # Unique (family, item) - an item can appear for several customers (shared accounts).
    unique = {(j["family"], j["item_id"]): (j, y) for j, y in labeled}
    pairs = list(unique.values())

    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    reliability = []
    for lo, hi in bins:
        inside = [(j["p"], y) for j, y in pairs if lo <= j["p"] < hi]
        if inside:
            reliability.append(
                {
                    "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                    "n": len(inside),
                    "mean_p": round(statistics.mean(p for p, _ in inside), 3),
                    "observed_yes": round(sum(y for _, y in inside) / len(inside), 3),
                }
            )
    brier = round(statistics.mean((j["p"] - y) ** 2 for j, y in pairs), 4) if pairs else None

    cutoffs = []
    for yes, no in CUTOFFS:
        correct = unsure = wrong = 0
        for j, y in pairs:
            if j["p"] >= yes:
                correct += y
                wrong += not y
            elif j["p"] <= no:
                correct += not y
                wrong += y
            else:
                unsure += 1
        cutoffs.append(
            {"yes": yes, "no": no, "correct": correct, "unsure": unsure, "wrong": wrong}
        )

    per_family: dict[str, dict] = {}
    for j, y in pairs:
        f = per_family.setdefault(j["family"], {"items": 0, "positives": 0})
        f["items"] += 1
        f["positives"] += y
    errors = [
        {"family": j["family"], "item": j["ref"], "p": j["p"], "label": y}
        for j, y in pairs
        if (j["p"] >= 0.7 and not y) or (j["p"] <= 0.3 and y) or 0.3 < j["p"] < 0.7
    ]
    return {
        "items": len(pairs),
        "brier": brier,
        "per_family": per_family,
        "reliability": reliability,
        "cutoffs": cutoffs,
        "errors_or_unsure_at_0.7_0.3": errors,
    }


def stability(runs: list[list[dict]]) -> dict:
    by_key: dict[tuple, list[str]] = {}
    for run in runs:
        for j in run:
            by_key.setdefault((j["customer_id"], j["id"]), []).append(j["band"])
    multi = [v for v in by_key.values() if len(v) == len(runs)]
    flips = sum(1 for v in multi if len(set(v)) > 1)
    return {"questions": len(multi), "band_changed": flips}


FAMILY_FIELD = {
    "ticket_unresolved_concern": "support_tickets",
    "alert_unexplained_risk": "alerts",
    "decision_open_condition": "decision_history",
}


def run_hard(repeats: int, yes: float = 0.7, no: float = 0.3) -> None:
    items = json.loads((Path(__file__).resolve().parents[1] / "evals" / "hard_items.json").read_text())["items"]

    def judge(item: dict) -> dict:
        field = FAMILY_FIELD[item["family"]]
        state = {"request": {"type": "credit_limit_increase", "category": "credit"}, field: [item["item"]]}
        judged = jev_client.evaluate_judgments(state)
        j = next(x for x in judged["judgments"] if x["family"] == item["family"])
        return {"id": item["id"], "p": j["p"], "band": j["band"], "model": judged["model"]}

    runs = []
    for _ in range(repeats):
        with ThreadPoolExecutor(max_workers=8) as pool:
            runs.append(list(pool.map(judge, items)))
    rows, correct, wrong, unsure_ok = [], 0, 0, 0
    for k, item in enumerate(items):
        ps = [run[k]["p"] for run in runs]
        bands = [run[k]["band"] for run in runs]
        b = bands[0]
        label = item["label"]
        if label is None:
            ok = b == "unsure"
            unsure_ok += ok
        else:
            ok = (b == "yes") == label and b != "unsure"
        correct += ok and label is not None
        wrong += (label is not None) and b != "unsure" and (b == "yes") != label
        rows.append(
            f"| {item['id']} | {item['family'].split('_')[0]} | {label} | "
            + " / ".join(f"{p:.2f}" for p in ps)
            + f" | {b}{' (flips)' if len(set(bands)) > 1 else ''} | {'ok' if ok else 'MISS'} |"
        )
    decided = sum(1 for i in items if i["label"] is not None)
    ambiguous = len(items) - decided
    md = "\n".join(
        [
            f"# Jev judgments on held-out hard items - {datetime.now():%Y-%m-%d %H:%M}",
            "",
            f"{len(items)} items, {repeats} runs each, model {sorted({r['model'] for r in runs[0]})}, bands yes >= {yes}, no <= {no}.",
            "",
            f"Labeled items: {correct}/{decided} correct, {wrong} confidently wrong, "
            f"{decided - correct - wrong} unsure (escalated). Ambiguous items flagged unsure: {unsure_ok}/{ambiguous}.",
            "",
            "| item | family | label | p per run | band (run 1) | |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    ) + "\n"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"judgments_hard_{stamp}.md").write_text(md)
    print(md)
    print(f"Saved evals/results/judgments_hard_{stamp}.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N customers")
    ap.add_argument("--hard", action="store_true", help="held-out hard items (evals/hard_items.json)")
    args = ap.parse_args()
    if args.hard:
        run_hard(args.repeats)
        return

    customers = customers_with_labeled_items()
    if args.limit:
        customers = customers[: args.limit]
    truth = load_item_labels()
    runs = []
    for r in range(args.repeats):
        print(f"run {r + 1}: judging {len(customers)} customers ...", flush=True)
        runs.append(run_once(customers))

    result = {
        # The served version, not the alias: thresholds belong to a model version.
        "model": sorted({j["model"] for j in runs[0] if j.get("model")}),
        "customers": len(customers),
        "score_run_1": score(runs[0], truth),
        "stability": stability(runs) if len(runs) > 1 else None,
    }
    s = result["score_run_1"]
    lines = [
        f"# Jev judgment calibration - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{s['items']} labeled items from {len(customers)} customers · Brier {s['brier']} · model {result['model']}",
        "",
        "Items per family (positives): "
        + ", ".join(f"{k} {v['items']} ({v['positives']})" for k, v in s["per_family"].items()),
        "",
        "## Reliability (predicted probability vs. observed yes-rate)",
        "",
        "| bin | n | mean p | observed yes |",
        "|---|---|---|---|",
        *[f"| {b['bin']} | {b['n']} | {b['mean_p']} | {b['observed_yes']} |" for b in s["reliability"]],
        "",
        "## Cut-offs",
        "",
        "| yes ≥ | no ≤ | correct | unsure | wrong |",
        "|---|---|---|---|---|",
        *[f"| {c['yes']} | {c['no']} | {c['correct']} | {c['unsure']} | {c['wrong']} |" for c in s["cutoffs"]],
    ]
    if result["stability"]:
        st = result["stability"]
        lines += ["", f"Stability across {args.repeats} runs: band changed on {st['band_changed']} of {st['questions']} questions."]
    if s["errors_or_unsure_at_0.7_0.3"]:
        lines += ["", "Errors or unsure at 0.7/0.3:", ""]
        lines += [f"- {e['family']} {e['item']}: p={e['p']} label={e['label']}" for e in s["errors_or_unsure_at_0.7_0.3"]]
    md = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"judgments_{stamp}.json").write_text(json.dumps(result, indent=1, default=str))
    (RESULTS_DIR / f"judgments_{stamp}.md").write_text(md)
    print(md)
    print(f"Saved evals/results/judgments_{stamp}.md")


if __name__ == "__main__":
    main()
