# Demo arc: from a tool-calling agent to "Jev judges, code decides"

A three-stage story you can run live with the toggle above the chat input, backed by evals on
labeled cases. Each stage keeps what worked in the previous one and fixes what the evals showed.

| Stage | Toggle | Who gathers evidence | Who decides |
|---|---|---|---|
| 1 | **1 · Claude + tools** | Claude picks graph tools itself | Claude |
| 2 | **2 · 10 queries + one Jev choice** | Code: 10 fixed Cypher retrievals | One Jev Choice (rules in its prompt); low confidence → Claude |
| 3 | **3 · 10 queries + Jev judgments + code** | Code: same 10 retrievals | Code guards over code facts + per-item Jev judgments; escalations → Claude with the written policy |

## The script

Use the same message in each mode:

```
Should we approve a credit limit increase for Tanya Trujillo? She is requesting a $4,000 limit increase.
```

Tanya passes every written rule: score 796, total limit 18% of income, no sanctions. In March
she was approved conditionally, on bringing an overdrawn account back to positive within 30
days. Nothing in the graph shows that happened. Under the bank's policy, the answer is "defer
until verified".

### Stage 1: let the LLM figure it out

*Claude + tools.* Claude chooses which graph tools to call (customer search, precedents,
policies, fraud patterns...) and recommends.
- **Show:** the tool-call list in the answer, and the graph filling in.
- **Point:** it works and it's flexible. But each run chooses its own evidence, so the same
  question can get different evidence and a different answer.
- **History:** in September this path took 60-100 s and seven tool calls, and flipped between
  deny and conditional approve on Jessica Norris (`docs/decision-sop-with-jev.md` §1, §5). It
  also missed evidence it never retrieved: a sanctioned co-owner, a frozen account.
- **Today, on the labeled cases** (`eval_20261112_130658.md`, stage-1 tool set): 14.5 s with
  Claude Sonnet 5.5, 4.6 tool calls per case, and 10/14 final answers matched policy. It
  approved both unmet-condition cases, Tanya included, though it read her decision history.

### Stage 2: watch which tools mattered, then run them every time

We watched which tools the agent actually used and found useful. That became a fixed plan of
10 Cypher retrievals that code runs every time: customer, account parties with sanctions
status, notable transactions, transaction summary, tickets, alerts, decision history,
precedents, base rates, policies. They run concurrently; retrievals plus Jev take about 0.5 s.

Then one Jev Choice, `yes / no / needs_llm_decision`, reads all of it and decides, with the
business rules written into the Choice's criteria.
- **Show:** the fast card, "yes 90%", in under a second.
- **Point:** same evidence every time, and an answer in under a second. In September, Jev's
  verdicts agreed 20/20 with Claude's on the same evidence. So Jev made the answer faster and
  cheaper, but didn't change what was decided.
- **But:** on Tanya it says **yes**. All of the rule-checkable facts look clean, and the one
  sentence that matters is buried in a prior decision's text. The confidence (0.85) gives no
  hint.

### Stage 3: break up the monolith

TypeSafe's own guidance: *"Jev makes judgments, code makes decisions."*
- **Code computes** everything that is a lookup or arithmetic: thresholds, sanctions, open
  cases, approval tier.
- **Jev answers one narrow question per item that needs reading.** Is this ticket an
  unresolved concern? Did this prior decision leave a condition unmet? Is this alert
  unexplained? Is this precedent comparable?
- **Ordered code guards decide.** Only written rules can reject. A Jev judgment can only
  escalate.
- **Escalations go to Claude** with the bank's written policy for the specific guard that fired.

- **Show:** the card escalates with `g_open_condition`. It cites decision 88cb6990 at p=0.94
  and shows the policy text handed to Claude. Claude checks the evidence and defers: approve
  once the condition is verified.
- **Point:** the answer names the rule and the item. If it's wrong, you can tell whether the
  judgment or the rule was wrong.

## What the evals say

All results are in `backend/evals/results/`. The labeled cases are 14 planted customers (7
scenarios × 2) who pass every written rule; only the narrative text should change the
outcome. Expected answers follow the written escalation policy
(`decision_policy.ESCALATION_POLICY`, v2026-11-11).

**Final answers vs. policy** (`eval_20261111_123713.md`, Claude Sonnet 5.5):

| | 1 · Claude + tools | 10 queries + Claude | 10 queries + Claude + **written policy** | 2 · one Jev choice | 3 · decomposed |
|---|---|---|---|---|---|
| correct finals | 11/14* | 10/14 | **14/14** | 9/14 | **14/14** |
| mean seconds | 14.1 | 10.4 | 8.5 | 15.5 | 9.9 |
| decided with no LLM call | 0 | 0 | 0 | 10/14 | **8/14** (~0.45 s) |
| extra tool calls | 4.9 | 0.1 | 0 | 0.5 | 0 |

\* This run could still call `compute_credit_figures` (denied). Rerun with the stage-1 tool set only: 10/14, 14.5 s, 4.6 calls (`eval_20261112_130658.md`).

**Routing before Claude** (labeled cases): stage 3 escalates exactly the 6 cases that should
escalate. Stage 2 waves through 2–3 of them, varying by run.

**How to read this honestly:**
1. **Accuracy came from writing the policy down, not from decomposing.** Claude with the
   whole written policy in its prompt also gets 14/14. Before the policy existed, Claude's
   calls on escalations split approve/reject even within identical scenario pairs
   (`eval_20261111_121815.md`).
2. **What decomposition buys** (with the written policy, accuracy and consistency are equal):
   - 57% of cases are decided in about 0.45 s with no LLM call.
   - Every decision has a trace naming the rule and the item.
   - Judgments can be scored one at a time: 211/211 on the template items, and 16/18 on the
     held-out hard items with zero confident errors.
   - The fast path fails closed.
   - Each new edge case is a new guard plus a question, not more prose in a prompt.
3. **Stage 2's misses are fixable with prompt prose.** Extending the Choice's criteria to name
   the planted edge cases got it to 14/14 routing. That is the "prompt gymnastics" TypeSafe
   warns about, and it was tuned on the same cases it was tested on.
4. **Consistency, 3 repeats** (`eval_20261111_125026.md`): both "10 queries + Claude + written
   policy" and stage 3 got 42/42 final answers right and gave the same answer on all 14 cases
   across repeats. Claude + policy is a little faster end to end (8.1 s vs 10.1 s), because
   stage 3's escalations make two passes. But stage 3 answered 24 of 42 runs in ~0.56 s with no
   LLM call.
5. **Small, synthetic, self-written.** 14 cases, labels I wrote, scenarios designed alongside the
   Jev questions. These are directional numbers for a demo, not a benchmark.

## Is this best practice? Assessment against current sources

The field is new and most numbers in these sources are vendor-reported.

| Practice | Source | Our stage 3 |
|---|---|---|
| Code owns rules, lookups and arithmetic; the model owns narrow judgments | [TypeSafe blog](https://typesafe.ai/blog/dont-call-it-a-decision-model); [Strands Decider example](https://github.com/strands-labs/strands-decider/blob/main/examples/strands/tool_call_intervention.py) ("the model classifies, the code decides"); [AWS prescriptive guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-security/best-practices-system-design.html) | ✅ |
| Atomic questions, many in one request | [TypeSafe build guide](https://docs.typesafe.ai/concepts/how-to-build-with-system-one), [fan-out](https://docs.typesafe.ai/patterns/fan-out) | ✅ one Noul per item, one request |
| Never let one model answer authorise an irreversible action | [Superteams](https://www.superteams.ai/blog/decision-models-explained-jev-cloudflare-clef-and-the-system-one-wave/), [AWS Industries](https://aws.amazon.com/blogs/industries/is-your-ai-agent-ready-for-prime-time/) | ✅ only written rules reject; Jev only escalates |
| Middle band gets its own route; riskier actions get stricter cut-offs | [TypeSafe probability routing](https://docs.typesafe.ai/patterns/probability-routing), [noul screen](https://docs.typesafe.ai/patterns/noul-screen) | ✅ unsure escalates. "No" cut-off tightened to 0.2 for the approve path, from held-out items |
| Measure thresholds on your own labels; check calibration | TypeSafe [confidence](https://docs.typesafe.ai/confidence); Superteams ("never trust a threshold you did not measure") | ⚠️ started: `scripts/eval_judgments.py` reliability table plus held-out hard items. Far below the suggested 300-500 real labeled decisions |
| Pin model versions; retune thresholds when the version changes | Superteams; [Waehner](https://www.kai-waehner.de/blog/2026/09/28/how-system-one-models-like-jev-change-enterprise-ai-architecture/) | ⚠️ the version is recorded (`jev-1.14.0`) and noted in config, but `jev-latest` is still the alias in use |
| Three lanes: auto / LLM second opinion / human | Superteams; Strands Decider README; AWS Industries | ⚠️ auto + LLM only. No human lane yet. `senior_review` should go to a person |
| The LLM is the escalation path, applying written policy, not its own appetite | Waehner; Strands ("code decides") | ✅ per-guard policy handed to Claude. Stronger version: Claude answers the guard's open question and code re-runs the guards |
| Fail closed on model errors | Strands interventions (`onError: deny`) | ✅ an unanswered judgment counts as unsure and escalates |
| Log probabilities and outcomes; track escalation and reversal rates | Strands, AWS Industries, Waehner | ⚠️ the trace is in every verdict and the eval JSON, but not persisted as a decision log |
| Treat state text as untrusted (prompt injection) | Superteams | ⚠️ not addressed: ticket text goes to Jev and Claude as-is |
| Compute dates and windows in code | [TypeSafe model jaggedness](https://docs.typesafe.ai/model-jaggedness) ("Jev is not a calculator") | ⚠️ the condition question compares records by text; code should compute "days since condition" |
| Avoid an `other` option in a Choice; use a companion Noul | [TypeSafe companion-noul](https://docs.typesafe.ai/patterns/companion-noul) | stage 2's `needs_llm_decision` is exactly that; stage 3 has no Choice |

**Verdict:** stage 3 is the architecture the sources recommend, and stage 2 is the
anti-pattern they name. The open gaps are operational rather than architectural: real labeled
data and calibration, a human lane, persistent decision logs, version pinning, date handling
in code, and treating text as untrusted.

Related systems worth a mention in the talk:
- [Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/) is a Jev-compatible
  decision model with calibration-oriented training. It's a possible fallback or second opinion.
- [Strands Decider](https://github.com/strands-labs/strands-decider) (AWS Strands Labs,
  experimental) is an open-weights analogue with the same noul/choice/score shapes, used with
  Strands' intervention handlers.

## Commands

From the repo root, in this order:

```bash
make demo-setup      # Neo4j up (docker compose), generate data, plant 14 labeled customers, embeddings
make dev             # backend :8000 + frontend :3000
make eval-fast       # routing on the labeled cases, no Claude (~1 min)
make eval            # all arms incl. Claude (~15 min)
make eval-judgments  # Jev calibration: graph items + held-out hard items
```

- `make demo-setup` clears and regenerates the database. To refresh without regenerating, run
  `make plant-scenarios embeddings`. Both are idempotent.
- Claude: if the `ANTHROPIC_API_KEY` in `.env` is invalid, add `CLAUDE_LOGIN=1`
  (e.g. `make dev CLAUDE_LOGIN=1`, `make eval CLAUDE_LOGIN=1`) to use the Claude CLI login instead.
  `CLAUDE_MODEL` defaults to `claude-sonnet-5-5`.
