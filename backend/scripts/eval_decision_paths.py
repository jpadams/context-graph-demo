"""
Eval: four arms on the same (customer, amount) cases, each differing by one thing.

  original         Claude with the original graph tools picks its own Cypher, then recommends.
  derived_tool     Same tool loop plus a compute_credit_figures tool (isolates pre-computation).
  evidence_claude  Code runs the fixed evidence plan; Claude decides from it; no Jev
                   (isolates code-owned evidence gathering).
  fast_answer      Fixed evidence plan -> Jev typed verdict -> Claude explains or, when
                   escalated, decides (the production path).

Records the final recommendation, confidence, wall time, tool calls, Jev verdict and
evidence coverage per arm, then writes JSON + a Markdown summary to evals/results/.
Select arms with --arms; merge rows from an earlier results file with --merge so arms
run at different times can be compared on the same cases.

    python scripts/eval_decision_paths.py --sample 10 --amounts 5000,25000
    python scripts/eval_decision_paths.py --sample 10 --amounts 5000,25000 \\
        --arms derived_tool,evidence_claude --merge evals/results/eval_20260921_210307.json
    python scripts/eval_decision_paths.py --customers "Jessica Norris" --arms fast_answer

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

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"
ARMS = ["original", "derived_tool", "evidence_claude", "fast_answer"]
TOOL_LOOP_ARMS = {"original": ("compute_credit_figures",), "derived_tool": ()}


def sample_customers(n: int, seed: int = 7) -> list[str]:
    rows = context_graph_client.execute_cypher(
        """
        MATCH (p:Person)-[:OWNS]->(:Account)
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


async def run_case(customer: str, amount: float, repeat: int, arms: list[str]) -> dict:
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
    }

    for arm in arms:
        print(f"  [{arm:15s}] {resolved['name']} ${amount:,.0f} (repeat {repeat}) ...", flush=True)
        started = time.perf_counter()
        try:
            if arm in TOOL_LOOP_ARMS:
                run = await run_agent_only(message, exclude_tools=TOOL_LOOP_ARMS[arm])
            elif arm == "evidence_claude":
                case = await asyncio.to_thread(
                    collect_credit_line_increase_evidence, resolved["id"], amount
                )
                run = await run_claude_on_evidence(message, case)
            else:  # fast_answer
                fa = await asyncio.to_thread(run_credit_line_increase, resolved["id"], amount)
                explanation = await run_fast_answer_explanation(message, fa)
                v = fa["verdict"]
                run = {
                    **explanation,
                    "jev_decision": v["decision"],
                    "jev_routed": v["routed_decision"],
                    "jev_confidence": v["confidence"],
                    "jev_probabilities": v["probabilities"],
                    "jev_signals": v["signals"],
                    "fast_answer_ms": fa["total_ms"],
                    "retrievals": {r["name"]: r["count"] for r in fa["retrievals"]},
                    "seconds": round(time.perf_counter() - started, 1),
                }
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
    fast = [
        r["runs"]["fast_answer"]
        for r in rows
        if "jev_routed" in r.get("runs", {}).get("fast_answer", {})
    ]
    if fast:
        summary["jev"] = {
            "decided_independently": sum(
                1 for f in fast if f["jev_routed"] != "needs_llm_decision"
            ),
            "escalated": sum(1 for f in fast if f["jev_routed"] == "needs_llm_decision"),
            "mean_confidence": round(statistics.mean(f["jev_confidence"] for f in fast), 2),
        }
    return summary


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
    if "jev" in summary:
        j = summary["jev"]
        out += [
            "",
            f"Jev decided independently in {j['decided_independently']} runs, escalated {j['escalated']}, mean confidence {j['mean_confidence']}.",
        ]
    out += ["", "Pairwise agreement on the final recommendation:", ""]
    for pair, v in summary.get("pairwise_agreement", {}).items():
        out.append(f"- {pair}: {v['agree']}/{v['compared']}")
    out += [
        "",
        "## Per run",
        "",
        "Cell format: final (confidence) seconds / tool calls.",
        "",
        "| customer | amount | original | derived_tool | evidence_claude | jev → routed (conf) | fast_answer |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(run):
        if not run or "error" in run:
            return "-"
        return f"{run.get('final', '-')} ({run.get('confidence', '-')}) {run.get('seconds', '-')}s / {run.get('tool_call_count', '-')}"

    for r in rows:
        runs = r.get("runs", {})
        f = runs.get("fast_answer", {})
        out.append(
            f"| {r.get('customer')} | ${r.get('amount', 0):,.0f} | {cell(runs.get('original'))} | "
            f"{cell(runs.get('derived_tool'))} | {cell(runs.get('evidence_claude'))} | "
            f"{f.get('jev_decision', '-')} → {f.get('jev_routed', '-')} ({f.get('jev_confidence', '-')}) | {cell(f)} |"
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
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if args.skip_original:
        arms = [a for a in arms if a != "original"]

    customers = [c.strip() for c in args.customers.split(",") if c.strip()]
    if args.sample:
        customers += sample_customers(args.sample)
    if not customers:
        customers = ["Jessica Norris"]
    amounts = [float(a) for a in args.amounts.split(",")]

    rows = []
    for customer in customers:
        for amount in amounts:
            for repeat in range(1, args.repeats + 1):
                rows.append(await run_case(customer, amount, repeat, arms))

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
