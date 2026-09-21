# Design: AI decisions on a context graph as a standard operating procedure

This document records how the "fast answer" decision path works, the choices behind it,
and what we learned building it. It started from one demo question, *"Should we approve a
credit limit increase for Jessica Norris? She's requesting a $25,000 limit increase."*, and
generalised into a parameterized case that runs for any customer and amount.

## 1. The problem with the original path

The original demo answered approval questions with a tool-use agent: Claude read the question,
picked graph tools one at a time (search customer, find precedents, get policy, detect fraud
patterns...), read each result, and eventually recommended. Observed behaviour on the same
question across runs:

- **Slow.** 100+ seconds and 7 tool calls; every tool call is a full model round trip.
- **Inconsistent.** Which evidence got gathered depended on which tools Claude chose that
  run. The same question produced "deny" one day and "conditional approve" the next.
- **Contaminated.** Tool results carried community-neighbourhood data (other people's
  accounts, counterparties) that the agent sometimes attributed to the customer.
- **Silently incomplete.** Precedent search returned nothing for weeks because no
  embeddings existed and the batch-embedding endpoint was broken; nothing surfaced this.

## 2. The approach: code owns the SOP, models supply judgment

A decision **case** is a standard operating procedure fixed in code:

| Part | What it is | Where |
|---|---|---|
| **Slots** | Parameters the case needs: customer, amount, category, request type | `backend/app/cases.py` |
| **Evidence plan** | A fixed, named set of Cypher retrievals run in code, each reporting row count, status and latency | `backend/app/evidence.py` (`EVIDENCE_RETRIEVALS`, `collect_evidence`) |
| **Typed questions** | A Jev Choice for the verdict plus Nouls for signals, one per precedent and per policy | `backend/app/jev_client.py` (`build_decision_questions`) |
| **Routing policy** | Confidence floor and escalation rule applied in code | `jev_client.evaluate_decision`, `JEV_CONFIDENCE_FLOOR` |
| **Explanation / final call** | Claude receives the same evidence and verdict; explains, or decides when escalated | `backend/app/agent.py` (`format_fast_answer_context`) |

Two answers come back from one question:

1. **Fast answer (about 1 to 5 s).** Evidence plan -> Jev typed verdict -> UI card. Jev is
   TypeSafe's System One model: it returns typed answers with calibrated probabilities in one
   pass and generates no text, so it costs cents per thousand decisions and cannot hallucinate
   a narrative.
2. **Regular answer (about 20 s).** Claude explains the verdict from the identical evidence.
   It may still call tools, and the UI reports exactly which ones it used. When Jev escalated,
   Claude makes the final approve/reject call.

The chat path fills the slots from free text (Jev reads the intent and category, code
resolves the customer name against Person nodes and parses the amount). The API
(`POST /api/cases/credit-line-increase`) and the eval harness fill them directly.

## 3. Choices and why

**Code runs the Cypher, not the LLM.** Same evidence every run, sub-second first answer,
and a visible list of what was gathered. Claude can still reach for tools, but that becomes
an observable exception rather than the mechanism.

**One Choice with three options: `yes`, `no`, `needs_llm_decision`.** The third option is a
first-class outcome, not a fallback. Jev picks it when evidence is mixed or a policy requires
case-by-case review. We first called it `needs_human_decision`; we changed it to escalate to
Claude so the demo produces a final recommendation, and because a reasoning model is the
natural next tier when a System One model is unsure. A human tier could sit above it.

**A confidence floor in code (default 0.6).** Jev's Choice confidence summarises how
concentrated its probabilities are. Below the floor, even a `yes` or `no` is escalated. This
follows TypeSafe's confidence-gated routing pattern; thresholds are a business setting, not a
model property, so they live in config.

**Ask independent Nouls in the same call.** Signals (elevated risk, manual review required,
evidence sufficient, sanctioned party exposure, open fraud activity) make the verdict
explainable and are reusable by code. They run in parallel with the verdict at negligible
extra latency.

**One question per precedent and per policy.** An aggregate "precedents favour approval"
Noul returned 85% while the precedents list was empty (it was reading two unrelated
approvals in the customer's history). Per-item questions ("does `precedents[2]` support
approving this request?") give an inspectable trace: the UI lists each precedent as *for* or
*against* with its probability, and the aggregate is computed in code as the mean.

**Never ask about empty evidence.** If a retrieval returned no rows, the dependent question
is not asked and the signal is `null`, rendered as "no evidence". Jev is trained for
calibrated decisions, not for noticing that a list is empty; code knows the count.

**Faithful evidence, scoped to the customer.** Retrievals follow explicit edges from the
Person node: owned accounts, organisations that co-own or transact with those accounts
(with sanctions status), notable and summarised transactions, support tickets, alerts,
decision history, precedents by reasoning-embedding similarity, category base rates,
policies. Adding the co-owner and ticket retrievals flipped the Jessica Norris case from
approve to reject: her account is co-owned by a watch-listed organisation and she has an
open fraud report. Neither was in the first evidence plan.

**A parseable final line from Claude.** Claude ends with
`FINAL_RECOMMENDATION: approve|reject (confidence x)`. Code parses it, relates it to Jev's
verdict (concurs / disagrees / decided on escalation), and the UI shows it as the final
answer. The line is stripped from the displayed prose.

**Report what Claude needed beyond the plan.** The `done` event carries
`additional_tools_used`. In every observed run of the credit case Claude needed nothing extra,
which is itself evidence that the evidence plan is adequate.

## 4. Things we learned about the data and the models

- Jev's verdicts on this case are stable in direction but land near the floor (confidence
  0.3 to 0.5), and its `evidence_sufficient` signal sits below 0.4. The sample Person nodes
  carry no income or credit score, while the Credit Limit Policy is defined by both. Jev is
  correctly saying the case is under-specified. Adding those fields to the generator would
  likely produce confident verdicts.
- Precedent retrieval is one-sided: semantic search on a question containing "approve"
  returns approvals. A balanced retrieval (top approvals, rejections and escalations) would
  give Jev contrast. Not done yet.
- Jev intent-read latency varied from 0.5 s to 3 s; the OpenAI embedding call for
  precedents takes 2 to 3 s. Retrievals run sequentially today; running them concurrently
  and overlapping the intent read would bring the fast answer back near 1.5 s.
- Neo4j 2026.01 rejects aggregation inside map projections. Two queries in
  `context_graph_client.py` were rewritten with a preceding `WITH`.
- The `sanctioned_party_exposure` question also fires on poor risk ratings with clear
  sanctions status. Splitting it into two Nouls would be cleaner.

## 4b. Giving Jev enough to decide (2026-09-21)

The first eval showed Jev escalating every case with `evidence_sufficient` near 0.18: the
Person nodes had no income or credit score while the Credit Limit Policy was defined by both.
Changes:

- The generator now gives each Person `credit_score`, `annual_income`, `income_verified`,
  `employment_status`, `current_credit_limit` and `credit_utilization`, loosely
  anti-correlated with `risk_score`. `generate_sample_data.py --enrich-only` adds them to an
  existing database without regenerating it.
- The Credit Limit Policy and Exception Approval Authority carry explicit rules in
  `threshold_rules` and in their description text (minimum score 660, total limit at most
  35% of income, a single increase at most 100% of the current limit, senior review above
  $50,000; manager / senior / committee approval tiers).
- Code computes the policy arithmetic and puts it in `request.derived`: proposed total limit,
  its share of income, and the increase relative to the current limit. Known calculations stay
  in code; Jev judges, it does not do arithmetic.

Effect on Jessica Norris ($25,000 on a $5,000 limit, $28,500 income): Jev answers `no` at
0.94 confidence with no escalation, `violates_policy` 0.95, `evidence_sufficient` 0.69.
At $5,000 it still answers `no` (0.66), because the total would exceed 35% of income and the
sanctions and fraud signals remain.

## 5. Evaluation

`backend/scripts/eval_decision_paths.py` runs the same (customer, amount) cases through up to
four arms, with repeats, and records final recommendation, confidence, wall time, tool calls,
Jev verdict and evidence coverage. It reports per-arm consistency across repeats and pairwise
agreement, and writes JSON plus Markdown to `backend/evals/results/`.

The first two evals compared whole pipelines, which vary three things at once: who gathers
the evidence, what the evidence contains, and who decides. The four arms isolate them:

| arm | gathers evidence | evidence contains | decides |
|---|---|---|---|
| `original` | Claude via the original tools | raw fields, policy text | Claude (Sonnet 4.5 by default) |
| `derived_tool` | Claude via tools + `compute_credit_figures` | raw fields + code-computed ratios on request | Claude |
| `evidence_claude` | code, fixed plan | full plan incl. `request.derived` | Claude, no Jev |
| `fast_answer` | code, fixed plan | full plan incl. `request.derived` | Jev, Claude on escalation |

`original` vs `derived_tool` isolates pre-computation; `derived_tool` vs `evidence_claude`
isolates code-owned gathering; `evidence_claude` vs `fast_answer` isolates what Jev adds over
Claude reading identical evidence. `--merge` combines arms run at different times on the same
cases. The `original` arm excludes tools added after the original demo
(`POST_ORIGINAL_TOOLS` in `agent.py`) so it stays reproducible as the tool set grows.

```bash
cd backend && set -a && source ../.env && set +a
.venv/bin/python scripts/eval_decision_paths.py --customers "Jessica Norris,Katherine Miller" --amounts 25000,60000 --repeats 2
.venv/bin/python scripts/eval_decision_paths.py --sample 5 --skip-original   # fast path only
```

### First results (2026-09-21, 3 customers x $25,000 x 2 repeats)

| path | mean s | mean tool calls | finals | consistent across repeats |
|---|---|---|---|---|
| original (tool loop) | 63.5 | 5.8 | approve 4, reject 2 | 3/3 cases |
| fast answer (fixed Cypher + Jev + Claude) | 32.7 | 0.2 | reject 6 | 3/3 cases |

Cross-path agreement: 2 of 6 runs (both John Walsh runs). Full table in
`backend/evals/results/eval_20260921_170153.md`.

What the detail shows:

- Both paths were internally consistent on this small sample; the tool-loop path chose the
  same tools each time for a given customer. The earlier flip-flop on Jessica Norris came from
  missing embeddings, not from tool selection alone.
- The paths **disagreed on the same facts**. Both saw Jessica Norris's watch-listed account
  co-owner and open fraud report. The tool-loop path said "conditional approve, pending
  verification"; the fast path said "reject until resolved". Same for Katherine Miller's open
  escalated alert. The difference is the decision posture, not the evidence.
- Jev **escalated every case** (confidence 0.30 to 0.44) and Claude rejected every escalation.
  Two contributors: the sample data lacks income and credit score, so `evidence_sufficient`
  sits near 0.18 on every customer; and the escalation prompt tells Claude that missing
  information counts against approval. That is a defensible posture for credit, but it means
  the current setup is conservative by construction. A ground-truth label set is needed
  before calling either path "more accurate".
- The fast path is 2x faster end to end and needed an extra tool in 1 of 6 runs
  (one Cypher query on John Walsh's closed account).

Next eval steps: add income and credit score to the sample generator so Jev can decide
without escalating; label a handful of cases by hand; widen to 10+ customers and two amounts;
split `sanctioned_party_exposure` into sanctions vs. risk-rating questions.

### Second results (2026-09-21, after credit fields: 10 sampled customers x $5,000 and $25,000)

| path | mean s | median s | mean tool calls | finals |
|---|---|---|---|---|
| original (tool loop) | 68.3 | 47.5 | 5.3 | approve 9, reject 11 |
| fast answer (fixed Cypher + Jev + Claude) | 26.1 | 27.0 | 0.0 | approve 4, reject 16 |

- Jev decided independently in 14 of 20 cases (mean confidence 0.70, evidence_sufficient 0.66,
  up from 0.18) and escalated 6. Where Jev decided, the tool-loop path agreed 11 of 14 times.
  Overall agreement 13 of 20.
- The fast answer itself averaged 3.1 s (0.5 to 5.6 s); Claude's pass brought the total to 26 s.
  The tool loop had one 441 s outlier.
- Disagreements split two ways. At $25,000 the tool loop approved two customers whose proposed
  total exceeded 35% of income (Jev: policy violated 0.95); the fixed plan applies the written
  rule, the tool loop did not compute it. At $5,000, four escalations with low policy-violation
  probability still ended in Claude rejecting; the escalation prompt's "missing information
  counts against approval" posture is doing that, and it is the next thing to tune.

- Model tier check: the tool loop ran on the Agent SDK default (`claude-sonnet-4-5-20250929`;
  the UI label used to say sonnet-4, now fixed). Re-running the two $25,000 disagreements on
  Opus (`CLAUDE_MODEL=opus`) produced reject on both, with the model doing the income and
  increase arithmetic itself. The tool loop had the raw inputs all along (income and current
  limit from `search_customer`, rules from `get_policy`); it lacked the computation. Code-side
  `request.derived` removes the dependence on model tier for that step.

Full tables in `backend/evals/results/eval_20260921_210307.md`.

### Third results (2026-09-21, four arms on the same 20 cases)

| arm | mean s | tool calls | finals |
|---|---|---|---|
| original | 68.3 | 5.3 | approve 9, reject 11 |
| derived_tool | 40.1 | 6.5 | approve 9, reject 11 |
| evidence_claude | 22.4 | 0.0 | approve 4, reject 16 |
| fast_answer | 26.1 | 0.0 | approve 4, reject 16 |

Pairwise agreement: original vs derived_tool 16/20; derived_tool vs evidence_claude 15/20;
evidence_claude vs fast_answer 20/20.

- **Pre-computation** (derived_tool) fixed the two $25,000 arithmetic misses but the loop still
  approved two requests whose 100%-increase breach it had the numbers for. Delivering figures is
  not the same as applying rules.
- **Code-owned gathering** moved five cases, all tool-loop approvals the fixed plan rejected on
  evidence the loop never retrieved (sanctioned co-owner, frozen account with alerts, unverified
  income) or rules it read past.
- **Jev vs Claude on identical evidence** agreed on all 20 final recommendations. Jev's value is
  latency (~3 s vs ~22 s to an answer), cost, structured per-item signals, and that 14 of 20
  cases never needed the Claude pass. It did not change what was decided.

Next step suggested by this: evaluate written policy thresholds in code and pass every model a
pass/fail list (`policy_checks`), so models weigh exceptions instead of noticing violations.

Full tables in `backend/evals/results/eval_20260921_221040.md`.

## 6. Adding a new case

1. Decide the slots and write the canonical message (see `credit_line_increase_message`).
2. Add or reuse named retrievals in `EVIDENCE_RETRIEVALS`; every retrieval reports count and
   status, and failures degrade to empty evidence rather than aborting.
3. Write the Jev questions in the case's terms: a Choice when the answer is one of a fixed
   set, a Noul for a yes/no condition, a Score for a degree along described levels. Keep one
   judgment per question, reference state fields with backticks, and include a no-match or
   escalate option.
4. Gate every question on the evidence it needs; derive aggregates in code.
5. Set the routing policy (floor, which options escalate) and what Claude is asked to do on
   escalation.
6. Add the case to the eval harness and run it against the original path before relying on it.
