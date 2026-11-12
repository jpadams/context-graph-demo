"""
Eval: five arms on the same (customer, amount) cases, each differing by one thing.

  original         Claude with the original graph tools picks its own Cypher, then recommends.
  derived_tool     Same tool loop plus a compute_credit_figures tool (isolates pre-computation).
  evidence_claude  Code runs the fixed evidence plan; Claude decides from it; no Jev
                   (isolates code-owned evidence gathering).
  evidence_claude_policy  Same, plus the whole written escalation policy in Claude's prompt
                   (control: is the gain the decomposition, or just the written policy?).
  fast_answer      Fixed evidence plan -> one Jev Choice decides (monolithic) -> Claude
                   explains or, when escalated, decides.
  decomposed       Fixed evidence plan -> code facts + per-item Jev judgments -> code guards
                   decide -> Claude explains or, when escalated, decides (the default path).
                   fast_answer vs decomposed isolates *how* Jev is used.

--labeled runs the planted cases in evals/labeled_cases.json (see
scripts/plant_decision_scenarios.py), where narrative text decides the right route, and
reports route accuracy per fast arm. Per-item judgment accuracy is scored against
evals/narrative_labels.json and the planted items. --no-claude stops the fast arms at the
routed verdict (seconds, no Claude pass) for quick routing comparisons.

Records the final recommendation, confidence, wall time, tool calls, Jev verdict and
evidence coverage per arm, then writes JSON + a Markdown summary to evals/results/.
Select arms with --arms; merge rows from an earlier results file with --merge so arms
run at different times can be compared on the same cases.

    python scripts/eval_decision_paths.py --sample 10 --amounts 5000,25000
    python scripts/eval_decision_paths.py --sample 10 --amounts 5000,25000 \\
        --arms derived_tool,evidence_claude --merge evals/results/eval_20260921_210307.json
    python scripts/eval_decision_paths.py --customers "Jessica Norris" --arms fast_answer
    python scripts/eval_decision_paths.py --labeled --arms fast_answer,decomposed --no-claude

Needs the backend .env (Neo4j, OpenAI, Anthropic, TypeSafe) loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.agent import run_agent_only, run_claude_on_evidence, run_fast_answer_explanation
from app.cases import (
    collect_credit_line_increase_evidence,
    credit_line_increase_message,
    resolve_customer,
    run_credit_line_increase,
)
from app.context_graph_client import context_graph_client

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
RESULTS_DIR = EVALS_DIR / "results"
ARMS = [
    "original",
    "derived_tool",
    "evidence_claude",
    "evidence_claude_policy",
    "fast_answer",
    "decomposed",
]
FAST_ARMS = {"fast_answer": "monolithic", "decomposed": "decomposed"}
EXPECTED_ROUTE = {"approve": "yes", "reject": "no", "escalate": "needs_llm_decision"}
ROUTE_TO_FINAL = {"yes": "approve", "no": "reject"}
TOOL_LOOP_ARMS = {"original": ("compute_credit_figures",), "derived_tool": ()}


def sample_customers(n: int, seed: int = 7) -> list[str]:
    rows = context_graph_client.execute_cypher(
        """
        MATCH (p:Person)-[:OWNS]->(:Account)
        WHERE p.planted_scenario IS NULL
        WITH p, count(*) AS accounts
        RETURN p.name AS name ORDER BY p.name
        """
    )
    random.Random(seed).shuffle(rows)
    return [r["name"] for r in rows[:n]]


def _report(arm: str, r: dict) -> None:
    extra = ""
    if "jev_routed" in r:
        extra = f" jev {r.get('jev_decision')}→{r.get('jev_routed')} ({r.get('jev_confidence')}),"
    if "evidence_ms" in r:
        extra += f" evidence {r.get('evidence_ms')} ms,"
    if "fast_answer_ms" in r:
        extra += f" fast answer {r.get('fast_answer_ms')} ms,"
    print(
        f"               -> {r.get('final')} ({r.get('confidence')}) in {r.get('seconds')}s,"
        f"{extra} {r.get('tool_call_count')} tool calls {r.get('tool_calls') or ''}"
        + (f"  ERROR {r['error']}" if "error" in r else ""),
        flush=True,
    )


async def run_fast_arm(
    arm: str, resolved: dict, amount: float, message: str, explain: bool, started: float
) -> dict:
    fa = await asyncio.to_thread(
        run_credit_line_increase, resolved["id"], amount, FAST_ARMS[arm]
    )
    v = fa["verdict"]
    if explain:
        run = await run_fast_answer_explanation(message, fa)
    else:
        final = ROUTE_TO_FINAL.get(v["routed_decision"])
        run = {"path": arm, "final": final, "confidence": None, "tool_calls": [], "tool_call_count": 0}
    run.update(
        {
            "jev_decision": v["decision"],
            "jev_routed": v["routed_decision"],
            "fast_answer_ms": fa["total_ms"],
            "retrievals": {r["name"]: r["count"] for r in fa["retrievals"]},
            "seconds": round(time.perf_counter() - started, 1),
        }
    )
    if v.get("mode") == "decomposed":
        run.update(
            {
                "jev_confidence": None,
                "decided_by": v["decided_by"]["guard"],
                "fired": v["fired"],
                "judgments": [
                    {k: j.get(k) for k in ("family", "item_id", "ref", "p", "band")}
                    for j in v["judgments"]
                ],
                "jev_ms": v["latency_ms"],
                "question_count": v["question_count"],
            }
        )
    else:
        run.update(
            {
                "jev_confidence": v["confidence"],
                "jev_probabilities": v["probabilities"],
                "jev_signals": v["signals"],
                "jev_ms": v["latency_ms"],
                "question_count": v.get("question_count"),
            }
        )
    return run


async def run_case(
    customer: str,
    amount: float,
    repeat: int,
    arms: list[str],
    explain: bool = True,
    label: dict | None = None,
) -> dict:
    resolved = resolve_customer(customer)
    if not resolved:
        return {"customer": customer, "amount": amount, "repeat": repeat, "error": "not_found"}
    message = credit_line_increase_message(resolved["name"], amount)
    row: dict = {
        "customer": resolved["name"],
        "customer_id": resolved["id"],
        "amount": amount,
        "repeat": repeat,
        "message": message,
        "runs": {},
        **(
            {
                "scenario": label["scenario"],
                "expected_route": EXPECTED_ROUTE[label["expected_route"]],
                "expected_final": label.get("expected_final"),
            }
            if label
            else {}
        ),
    }

    for arm in arms:
        print(f"  [{arm:15s}] {resolved['name']} ${amount:,.0f} (repeat {repeat}) ...", flush=True)
        started = time.perf_counter()
        try:
            if arm in TOOL_LOOP_ARMS:
                run = await run_agent_only(message, exclude_tools=TOOL_LOOP_ARMS[arm])
            elif arm in ("evidence_claude", "evidence_claude_policy"):
                case = await asyncio.to_thread(
                    collect_credit_line_increase_evidence, resolved["id"], amount
                )
                run = await run_claude_on_evidence(
                    message, case, with_policy=arm == "evidence_claude_policy"
                )
            else:  # fast_answer (monolithic) or decomposed
                run = await run_fast_arm(arm, resolved, amount, message, explain, started)
        except Exception as e:  # keep the eval going
            run = {
                "path": arm,
                "error": str(e),
                "final": None,
                "confidence": None,
                "seconds": round(time.perf_counter() - started, 1),
                "tool_call_count": None,
            }
        row["runs"][arm] = run
        _report(arm, run)
    return row


def summarize(rows: list[dict]) -> dict:
    summary: dict = {}
    for path in ARMS:
        runs = [
            r["runs"][path]
            for r in rows
            if path in r.get("runs", {}) and "error" not in r["runs"][path]
        ]
        if not runs:
            continue
        by_case: dict[tuple, list] = {}
        for r in rows:
            run = r.get("runs", {}).get(path)
            if run and "error" not in run:
                by_case.setdefault((r["customer"], r["amount"]), []).append(run["final"])
        multi = [v for v in by_case.values() if len(v) > 1]
        summary[path] = {
            "runs": len(runs),
            "mean_seconds": round(statistics.mean(x["seconds"] for x in runs), 1),
            "median_seconds": round(statistics.median(x["seconds"] for x in runs), 1),
            "mean_tool_calls": round(statistics.mean(x["tool_call_count"] or 0 for x in runs), 2),
            "finals": dict(Counter(x["final"] for x in runs)),
            "missing_final": sum(1 for x in runs if not x["final"]),
            "cases_with_repeats": len(multi),
            "consistent_cases": sum(1 for v in multi if len(set(v)) == 1),
        }
    agreement: dict[str, dict] = {}
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1 :]:
            agree = total = 0
            for r in rows:
                fa = r.get("runs", {}).get(a, {}).get("final")
                fb = r.get("runs", {}).get(b, {}).get("final")
                if fa and fb:
                    total += 1
                    agree += fa == fb
            if total:
                agreement[f"{a} vs {b}"] = {"agree": agree, "compared": total}
    summary["pairwise_agreement"] = agreement
    for arm in FAST_ARMS:
        fast = [r["runs"][arm] for r in rows if "jev_routed" in r.get("runs", {}).get(arm, {})]
        if not fast:
            continue
        confidences = [f["jev_confidence"] for f in fast if f.get("jev_confidence") is not None]
        summary[f"jev_{arm}"] = {
            "decided_without_claude": sum(1 for f in fast if f["jev_routed"] != "needs_llm_decision"),
            "escalated": sum(1 for f in fast if f["jev_routed"] == "needs_llm_decision"),
            "mean_confidence": round(statistics.mean(confidences), 2) if confidences else None,
            "mean_fast_answer_ms": round(statistics.mean(f["fast_answer_ms"] for f in fast)),
        }
        labeled = [
            (r, r["runs"][arm])
            for r in rows
            if r.get("expected_route") and "jev_routed" in r.get("runs", {}).get(arm, {})
        ]
        if labeled:
            misses = [
                f"{r['scenario']} / {r['customer']}: expected {r['expected_route']}, got {run['jev_routed']}"
                for r, run in labeled
                if run["jev_routed"] != r["expected_route"]
            ]
            by_scenario: dict[str, list[int]] = {}
            for r, run in labeled:
                hit = by_scenario.setdefault(r["scenario"], [0, 0])
                hit[0] += run["jev_routed"] == r["expected_route"]
                hit[1] += 1
            summary[f"route_accuracy_{arm}"] = {
                "correct": len(labeled) - len(misses),
                "total": len(labeled),
                "by_scenario": {k: f"{v[0]}/{v[1]}" for k, v in by_scenario.items()},
                "misses": misses,
            }
    for arm in ARMS:
        labeled = [
            (r, r["runs"][arm])
            for r in rows
            if r.get("expected_final") and arm in r.get("runs", {}) and "error" not in r["runs"][arm]
        ]
        if labeled:
            by_scenario: dict[str, list[int]] = {}
            misses = []
            for r, run in labeled:
                hit = by_scenario.setdefault(r["scenario"], [0, 0])
                ok = run.get("final") == r["expected_final"]
                hit[0] += ok
                hit[1] += 1
                if not ok:
                    misses.append(
                        f"{r['scenario']} / {r['customer']}: expected {r['expected_final']}, got {run.get('final')}"
                    )
            summary[f"final_accuracy_{arm}"] = {
                "correct": len(labeled) - len(misses),
                "total": len(labeled),
                "by_scenario": {k: f"{v[0]}/{v[1]}" for k, v in by_scenario.items()},
                "misses": misses,
            }
    judgment = score_judgments(rows)
    if judgment:
        summary["judgment_accuracy"] = judgment
    return summary


# Map a family to the label key that is its ground truth.
FAMILY_LABEL = {
    "ticket_unresolved_concern": ("tickets", "unresolved_concern"),
    "alert_unexplained_risk": ("alerts", "unexplained_risk"),
    "decision_open_condition": ("decisions", "open_condition"),
}
PLANTED_TRUTH = {
    "routine": False,
    "resolved_concern": False,
    "closed_but_unresolved": True,
    "cleared_benign": False,
    "closed_but_unexplained": True,
    "condition_follow_up": False,
}


def load_item_labels() -> dict[str, dict[str, bool]]:
    """item_id -> truth per label group, from narrative_labels.json and the planted cases."""
    truth: dict[str, dict[str, bool]] = {"tickets": {}, "alerts": {}, "decisions": {}}
    path = EVALS_DIR / "narrative_labels.json"
    if path.exists():
        labels = json.loads(path.read_text())
        for family, (group, key) in FAMILY_LABEL.items():
            for item_id, label in labels.get(group, {}).items():
                truth[group][item_id] = bool(label[key])
    path = EVALS_DIR / "labeled_cases.json"
    if path.exists():
        for case in json.loads(path.read_text())["cases"]:
            for item in case["planted"]:
                group = {"ticket": "tickets", "alert": "alerts", "decision": "decisions"}[item["kind"]]
                if item["variant"] == "conditional_approval":
                    value = case["scenario"] == "condition_unmet"
                else:
                    value = PLANTED_TRUTH[item["variant"]]
                truth[group][item["item_id"]] = value
    return truth


def score_judgments(rows: list[dict]) -> dict:
    """Per-family accuracy of banded Jev judgments against item labels (unique items)."""
    truth = load_item_labels()
    seen: dict[tuple, dict] = {}
    for r in rows:
        for j in r.get("runs", {}).get("decomposed", {}).get("judgments", []) or []:
            if j["family"] in FAMILY_LABEL:
                seen[(j["family"], j["item_id"])] = j
    out: dict[str, dict] = {}
    for (family, item_id), j in seen.items():
        group, _ = FAMILY_LABEL[family]
        if item_id not in truth[group]:
            continue
        expected = truth[group][item_id]
        s = out.setdefault(family, {"items": 0, "correct": 0, "unsure": 0, "wrong": []})
        s["items"] += 1
        if j["band"] == "unsure":
            s["unsure"] += 1
        elif (j["band"] == "yes") == expected:
            s["correct"] += 1
        else:
            s["wrong"].append(f"{j['ref']}: p={j['p']} expected {expected}")
    return out


def to_markdown(rows: list[dict], summary: dict, args: argparse.Namespace) -> str:
    out = [
        f"# Decision path eval - {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Cases: {len(rows)} runs · amounts {args.amounts} · repeats {args.repeats} · arms {args.arms}"
        + (f" · merged {Path(args.merge).name}" if args.merge else ""),
        "",
        "## Summary",
        "",
        "| arm | runs | mean s | median s | mean tool calls | finals | consistent cases |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in ARMS:
        s = summary.get(path)
        if s:
            out.append(
                f"| {path} | {s['runs']} | {s['mean_seconds']} | {s['median_seconds']} | "
                f"{s['mean_tool_calls']} | {s['finals']} | {s['consistent_cases']}/{s['cases_with_repeats']} |"
            )
    for arm in FAST_ARMS:
        j = summary.get(f"jev_{arm}")
        if j:
            out += [
                "",
                f"{arm}: decided without Claude in {j['decided_without_claude']} runs, escalated "
                f"{j['escalated']}, mean fast answer {j['mean_fast_answer_ms']} ms"
                + (f", mean Choice confidence {j['mean_confidence']}" if j["mean_confidence"] is not None else "")
                + ".",
            ]
    if any(f"route_accuracy_{a}" in summary for a in FAST_ARMS):
        out += ["", "## Route accuracy on labeled cases", "", "| scenario | " + " | ".join(a for a in FAST_ARMS if f"route_accuracy_{a}" in summary) + " |", "|---|" + "---|" * sum(1 for a in FAST_ARMS if f"route_accuracy_{a}" in summary)]
        arms_present = [a for a in FAST_ARMS if f"route_accuracy_{a}" in summary]
        scenarios = list(summary[f"route_accuracy_{arms_present[0]}"]["by_scenario"])
        for sc in scenarios:
            out.append(f"| {sc} | " + " | ".join(summary[f"route_accuracy_{a}"]["by_scenario"].get(sc, "-") for a in arms_present) + " |")
        out.append("| **total** | " + " | ".join(f"**{summary[f'route_accuracy_{a}']['correct']}/{summary[f'route_accuracy_{a}']['total']}**" for a in arms_present) + " |")
        for a in arms_present:
            for miss in summary[f"route_accuracy_{a}"]["misses"]:
                out.append(f"- {a} miss: {miss}")
    final_arms = [a for a in ARMS if f"final_accuracy_{a}" in summary]
    if final_arms:
        out += ["", "## Final answer accuracy on labeled cases (written escalation policy)", "", "| scenario | " + " | ".join(final_arms) + " |", "|---|" + "---|" * len(final_arms)]
        for sc in summary[f"final_accuracy_{final_arms[0]}"]["by_scenario"]:
            out.append(f"| {sc} | " + " | ".join(summary[f"final_accuracy_{a}"]["by_scenario"].get(sc, "-") for a in final_arms) + " |")
        out.append("| **total** | " + " | ".join(f"**{summary[f'final_accuracy_{a}']['correct']}/{summary[f'final_accuracy_{a}']['total']}**" for a in final_arms) + " |")
        for a in final_arms:
            for miss in summary[f"final_accuracy_{a}"]["misses"]:
                out.append(f"- {a} miss: {miss}")
    if summary.get("judgment_accuracy"):
        out += ["", "## Per-item Jev judgment accuracy (decomposed)", "", "| family | items | correct | unsure | wrong |", "|---|---|---|---|---|"]
        for family, v in summary["judgment_accuracy"].items():
            out.append(f"| {family} | {v['items']} | {v['correct']} | {v['unsure']} | {len(v['wrong'])} |")
        for family, v in summary["judgment_accuracy"].items():
            for w in v["wrong"]:
                out.append(f"- {family} wrong: {w}")
    out += ["", "Pairwise agreement on the final recommendation:", ""]
    for pair, v in summary.get("pairwise_agreement", {}).items():
        out.append(f"- {pair}: {v['agree']}/{v['compared']}")
    out += [
        "",
        "## Per run",
        "",
        "Cell format: final (confidence) seconds / tool calls.",
        "",
        "| customer | amount | expected | original | derived_tool | evidence_claude | monolithic jev → routed (conf) | fast_answer | decomposed route (guard) | decomposed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(run):
        if not run or "error" in run:
            return "-"
        return f"{run.get('final', '-')} ({run.get('confidence', '-')}) {run.get('seconds', '-')}s / {run.get('tool_call_count', '-')}"

    for r in rows:
        runs = r.get("runs", {})
        f = runs.get("fast_answer", {})
        d = runs.get("decomposed", {})
        out.append(
            f"| {r.get('customer')} | ${r.get('amount', 0):,.0f} | {r.get('expected_route', '-')} | {cell(runs.get('original'))} | "
            f"{cell(runs.get('derived_tool'))} | {cell(runs.get('evidence_claude'))} | "
            f"{f.get('jev_decision', '-')} → {f.get('jev_routed', '-')} ({f.get('jev_confidence', '-')}) | {cell(f)} | "
            f"{d.get('jev_routed', '-')} ({d.get('decided_by') or '-'}) | {cell(d)} |"
        )
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--customers", default="", help="comma-separated names or ids")
    ap.add_argument(
        "--sample", type=int, default=0, help="sample N customers with accounts (seeded)"
    )
    ap.add_argument("--amounts", default="25000", help="comma-separated USD amounts")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--arms", default=",".join(ARMS), help="comma-separated subset of " + ",".join(ARMS)
    )
    ap.add_argument(
        "--merge", default="", help="earlier results JSON merged by (customer, amount, repeat)"
    )
    ap.add_argument("--skip-original", action="store_true", help="drop the original arm")
    ap.add_argument(
        "--labeled", action="store_true", help="run the planted cases in evals/labeled_cases.json"
    )
    ap.add_argument(
        "--no-claude", action="store_true", help="fast arms stop at the routed verdict (no Claude pass)"
    )
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if args.skip_original:
        arms = [a for a in arms if a != "original"]

    explain = not args.no_claude
    rows = []
    if args.labeled:
        labeled = json.loads((EVALS_DIR / "labeled_cases.json").read_text())["cases"]
        for case in labeled:
            for repeat in range(1, args.repeats + 1):
                rows.append(
                    await run_case(case["customer_id"], case["amount"], repeat, arms, explain, case)
                )

    customers = [c.strip() for c in args.customers.split(",") if c.strip()]
    if args.sample:
        customers += sample_customers(args.sample)
    if not customers and not args.labeled:
        customers = ["Jessica Norris"]
    amounts = [float(a) for a in args.amounts.split(",")]

    for customer in customers:
        for amount in amounts:
            for repeat in range(1, args.repeats + 1):
                rows.append(await run_case(customer, amount, repeat, arms, explain))

    if args.merge:
        prior = json.load(open(args.merge))
        by_key = {(r.get("customer"), r.get("amount"), r.get("repeat")): r for r in rows}
        for old in prior["rows"]:
            key = (old.get("customer"), old.get("amount"), old.get("repeat"))
            if key in by_key:
                for arm, run in old.get("runs", {}).items():
                    by_key[key]["runs"].setdefault(arm, run)
            else:
                rows.append(old)

    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"eval_{stamp}.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=2, default=str)
    )
    md = to_markdown(rows, summary, args)
    (RESULTS_DIR / f"eval_{stamp}.md").write_text(md)
    print("\n" + md)
    print(f"Saved evals/results/eval_{stamp}.json and .md")


if __name__ == "__main__":
    asyncio.run(main())
