"""
Claude Agent SDK integration with Context Graph tools.
Provides MCP tools for querying and updating the context graph.
"""

import json
import re
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from .config import config
from .context_graph_client import context_graph_client
from .evidence import derive_credit_figures
from .gds_client import gds_client
from .vector_client import vector_client


def slim_properties(props: dict) -> dict:
    """Remove large properties to reduce response size."""
    slim = {}
    for key, value in props.items():
        # Skip embedding vectors
        if key in ("fast_rp_embedding", "reasoning_embedding", "embedding"):
            continue
        # Truncate long strings
        if isinstance(value, str) and len(value) > 200:
            slim[key] = value[:200] + "..."
        # Limit list sizes
        elif isinstance(value, list) and len(value) > 10:
            slim[key] = value[:10]
        else:
            slim[key] = value
    return slim


def get_graph_data_for_entity(entity_id: str, depth: int = 2, limit: int = 30) -> dict:
    """Get graph visualization data centered on an entity."""
    try:
        graph_data = context_graph_client.get_graph_data(
            center_node_id=entity_id, depth=depth, limit=limit
        )
        # Build nodes list first
        nodes = [
            {
                "id": node.id,
                "labels": node.labels,
                "properties": slim_properties(node.properties),
            }
            for node in graph_data.nodes
        ]

        # Create set of node IDs for filtering relationships
        node_ids = {node["id"] for node in nodes}

        # Only include relationships where both nodes exist
        relationships = [
            {
                "id": rel.id,
                "type": rel.type,
                "startNodeId": rel.start_node_id,
                "endNodeId": rel.end_node_id,
                "properties": slim_properties(rel.properties),
            }
            for rel in graph_data.relationships
            if rel.start_node_id in node_ids and rel.end_node_id in node_ids
        ]

        return {
            "nodes": nodes,
            "relationships": relationships,
        }
    except Exception as e:
        print(f"Error getting graph data for entity {entity_id}: {e}")
        return {"nodes": [], "relationships": []}


# ============================================
# SYSTEM PROMPT
# ============================================

CONTEXT_GRAPH_SYSTEM_PROMPT = """You are an AI assistant for a financial institution with access to a Context Graph.

The Context Graph stores decision traces - the reasoning, context, and causal relationships behind every significant decision made in the organization. This enables you to:

1. **Find Precedents**: Search for similar past decisions to inform current recommendations
2. **Trace Causality**: Understand how past decisions influenced subsequent outcomes
3. **Record Decisions**: Create new decision traces with full reasoning context
4. **Detect Patterns**: Identify fraud patterns and entity duplicates using graph structure

## Key Concepts

**Event Clock vs State Clock**:
- Traditional systems store the "state clock" - what is true right now
- The Context Graph stores the "event clock" - what happened, when, and with what reasoning

**Decision Traces**:
- Every significant decision is recorded with full reasoning
- Risk factors, confidence scores, and applied policies are captured
- Causal chains show how decisions influenced each other

## Guidelines

When helping users:
1. **Always search for precedents** before making recommendations
2. **Explain your reasoning thoroughly** - this becomes part of the decision trace
3. **Cite specific past decisions** when they inform your recommendation
4. **Flag exceptions or escalations** that may be needed
5. **Consider both structural and semantic similarity** when finding related cases

## Fast answers from Jev

For approve/deny questions the system may hand you a **fast answer** produced before you run:
a fixed set of Cypher retrievals (customer, account parties with sanctions status, notable
transactions, transaction summary, support tickets, alerts, decision history, precedents, category
outcomes, policies) and a typed verdict from TypeSafe's Jev model - `yes`, `no` or
`needs_llm_decision` - with calibrated probabilities, per-precedent and per-policy judgments, and
supporting signals (a signal of null means that evidence was empty, so say "no precedents were
found" rather than quoting a number). When present:
- Treat the pre-collected evidence as already retrieved; do not re-run the same lookups.
- Explain the verdict for the user: cite the specific precedents, policies and risk signals
  in the evidence that support or complicate it, and state the probabilities plainly.
- If Jev decided `yes` or `no`, confirm it unless the evidence clearly contradicts it; if you
  disagree, say so and explain why.
- If Jev escalated with `needs_llm_decision`, **you make the final call**: recommend approve or
  reject, weighing the evidence and any policy exceptions. Call tools only if something specific
  is missing (say what and why).
- Always end your reply with exactly one line in this form, and nothing after it:
  `FINAL_RECOMMENDATION: approve (confidence 0.8)` or `FINAL_RECOMMENDATION: reject (confidence 0.7)`

You have access to tools that leverage both:
- **Semantic similarity** (text embeddings) - for matching by meaning
- **Structural similarity** (FastRP graph embeddings) - for matching by relationship patterns

This combination provides insights that are impossible with traditional databases."""


# ============================================
# MCP TOOLS
# ============================================


def merge_graph_data(graphs: list[dict], max_nodes: int = 50, max_rels: int = 75) -> dict:
    """Merge multiple graph data objects, removing duplicates and limiting size."""
    all_nodes = {}
    all_relationships = {}

    for graph in graphs:
        if not graph:
            continue
        for node in graph.get("nodes", []):
            if len(all_nodes) < max_nodes:
                all_nodes[node["id"]] = node
        for rel in graph.get("relationships", []):
            # Only include relationships where both nodes are in the graph
            if rel.get("startNodeId") in all_nodes and rel.get("endNodeId") in all_nodes:
                if len(all_relationships) < max_rels:
                    all_relationships[rel["id"]] = rel

    return {
        "nodes": list(all_nodes.values()),
        "relationships": list(all_relationships.values()),
    }


@tool(
    "search_customer",
    "Search for customers by name, email, or account number. Returns customer profiles with risk scores and related account counts.",
    {"query": str, "limit": int},
)
async def search_customer(args: dict[str, Any]) -> dict[str, Any]:
    """Search for customers in the context graph."""
    try:
        results = context_graph_client.search_customers(
            query=args["query"], limit=args.get("limit", 10)
        )
        # Include graph data for top customers (1 hop from each)
        graphs = []
        for customer in results[:3]:  # Limit to first 3 customers
            customer_id = customer.get("id")
            if customer_id:
                customer_graph = get_graph_data_for_entity(customer_id, depth=1)
                graphs.append(customer_graph)

        # Merge all graph data with size limits
        graph_data = merge_graph_data(graphs) if graphs else {"nodes": [], "relationships": []}

        response = {
            "customers": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error searching customers: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_customer_decisions",
    "Get all decisions made about a specific customer, including approvals, rejections, escalations, and exceptions.",
    {"customer_id": str, "decision_type": str, "limit": int},
)
async def get_customer_decisions(args: dict[str, Any]) -> dict[str, Any]:
    """Get decisions about a customer."""
    try:
        results = context_graph_client.get_customer_decisions(
            customer_id=args["customer_id"],
            decision_type=args.get("decision_type"),
            limit=args.get("limit", 20),
        )
        # Include graph data centered on the customer
        graph_data = get_graph_data_for_entity(args["customer_id"], depth=2)

        response = {
            "decisions": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting decisions: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_similar_decisions",
    """Find structurally similar past decisions using FastRP graph embeddings. 
    Returns decisions with similar influences, causes, and precidents as well as decisions about related accounts .""",
    {
        "decision_id": {"type": str, "description": "The internal decision ID (decision.id)"},
        "limit": {
            "type": int,
            "description": "Number of similar decisions to return",
            "default": 5,
        },
    },
)
async def find_similar_decisions(args: dict[str, Any]) -> dict[str, Any]:
    """Find similar decisions using FastRP embeddings."""
    try:
        decision_id = args["decision_id"]
        limit = int(args.get("limit", 10))

        similar_decisions = gds_client.find_similar_decisions(decision_id, limit=limit)

        # Include graph data centered on the decision
        graph_data = get_graph_data_for_entity(decision_id, depth=2)

        response = {
            "similar_decisions": similar_decisions,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error finding similar decisions: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_precedents",
    "Find precedent decisions that could inform the current decision. Uses both semantic similarity (meaning) and structural similarity (graph patterns).",
    {"scenario": str, "category": str, "limit": int},
)
async def find_precedents(args: dict[str, Any]) -> dict[str, Any]:
    """Find precedent decisions using hybrid search."""
    try:
        results = vector_client.find_precedents_hybrid(
            scenario=args["scenario"], category=args.get("category"), limit=args.get("limit", 5)
        )
        # Include graph data for the first precedent found
        graph_data = None
        if results and len(results) > 0:
            first_id = results[0].get("id") if isinstance(results[0], dict) else None
            if first_id:
                graph_data = get_graph_data_for_entity(first_id, depth=2)

        response = {
            "precedents": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error finding precedents: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_causal_chain",
    "Trace the causal chain of a decision - what caused it and what it led to. Useful for understanding decision impact and history.",
    {"decision_id": str, "direction": str, "depth": int},
)
async def get_causal_chain(args: dict[str, Any]) -> dict[str, Any]:
    """Get the causal chain for a decision."""
    try:
        results = context_graph_client.get_causal_chain(
            decision_id=args["decision_id"],
            direction=args.get("direction", "both"),
            depth=args.get("depth", 3),
        )
        # Include graph data centered on the decision
        graph_data = get_graph_data_for_entity(args["decision_id"], depth=3)

        response = {
            "causal_chain": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting causal chain: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "record_decision",
    "Record a new decision with full reasoning context. Creates a decision trace in the context graph that can be referenced by future decisions.",
    {
        "decision_type": str,
        "category": str,
        "reasoning": str,
        "customer_id": str,
        "account_id": str,
        "risk_factors": list,
        "precedent_ids": list,
        "confidence_score": float,
    },
)
async def record_decision(args: dict[str, Any]) -> dict[str, Any]:
    """Record a new decision in the context graph."""
    try:
        # Generate embedding for the reasoning
        reasoning_embedding = None
        try:
            reasoning_embedding = vector_client.generate_embedding(args["reasoning"])
        except Exception:
            pass  # Continue without embedding if it fails

        decision_id = context_graph_client.record_decision(
            decision_type=args["decision_type"],
            category=args["category"],
            reasoning=args["reasoning"],
            customer_id=args.get("customer_id"),
            account_id=args.get("account_id"),
            risk_factors=args.get("risk_factors", []),
            precedent_ids=args.get("precedent_ids", []),
            confidence_score=args.get("confidence_score", 0.8),
            reasoning_embedding=reasoning_embedding,
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "success": True,
                            "decision_id": decision_id,
                            "message": f"Decision recorded successfully with ID {decision_id}",
                        },
                        indent=2,
                    ),
                }
            ]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error recording decision: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "detect_fraud_patterns",
    """Analyze accounts or transactions for potential fraud patterns using graph structure analysis.
    Checks an account's proximity to flagged transactions as well as the prevalance of flagged transactions in the community of related accounts.""",
    {
        "account_id": {
            "type": str,
            "description": "The internal account ID (account.id), not the customer-facing account number (account.account_number)",
        },
        "neighbor_count": {
            "type": int,
            "description": "Number of example decisions to return from the community",
            "default": 5,
        },
    },
)
async def detect_fraud_patterns(args: dict[str, Any]) -> dict[str, Any]:
    """Detect fraud patterns using graph analysis."""
    try:
        neighbor_count = int(args.get("neighbor_count", 5))
        results = gds_client.detect_fraud_patterns(
            account_id=args.get("account_id"),
            neighbor_count=neighbor_count,
        )
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error detecting fraud patterns: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_decision_community",
    "Find decisions in the same community using Leiden community detection. Returns decisions that are structurally related through causal chains and precedent relationships.",
    {
        "decision_id": {"type": str, "description": "The internal decision ID (decision.id)"},
        "example_count": {
            "type": int,
            "description": "Number of example decisions to return from the community",
            "default": 5,
        },
    },
)
async def find_decision_community(args: dict[str, Any]) -> dict[str, Any]:
    """Find decisions in the same community using Leiden."""
    decision_id = args["decision_id"]
    try:
        example_count = int(args.get("example_count", 5))
        results = gds_client.get_decision_community(
            decision_id=args["decision_id"], example_count=example_count
        )

        # Include graph data centered on the decision
        graph_data = get_graph_data_for_entity(decision_id, depth=2)

        response = {
            "community_decisions": results,
            "graph_data": graph_data,
        }
        return {"content": [{"type": "text", "text": json.dumps(response, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error finding community: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "find_accounts_with_high_shared_transaction_volume",
    "Find accounts that share high transaction volumes with a given account.",
    {
        "account_id": {
            "type": str,
            "description": "The internal account ID (account.id), not the customer-facing account number (account.account_number)",
        },
    },
)
async def find_accounts_with_high_shared_transaction_volume(args: dict[str, Any]) -> dict[str, Any]:
    """Find accounts with high shared transaction volume."""
    try:
        results = gds_client.find_accounts_with_high_shared_transaction_volume(
            account_id=args.get("account_id")
        )

        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error finding accounts with high shared transaction volume: {str(e)}",
                }
            ],
            "is_error": True,
        }


@tool(
    "compute_credit_figures",
    """Compute the arithmetic the Credit Limit Policy is written in, for a proposed credit limit increase:
    proposed total limit, its share of the customer's verified annual income, and the increase as a fraction
    of the current limit. Also returns the customer's credit_score, annual_income, current_credit_limit and
    credit_utilization. Use before judging a credit request against policy thresholds.""",
    {
        "customer_id": {"type": str, "description": "The internal customer ID (person.id)"},
        "requested_amount": {"type": float, "description": "Requested increase in USD"},
    },
)
async def compute_credit_figures(args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic policy arithmetic, so the model does not have to do division."""
    try:
        customer = context_graph_client.get_customer(args["customer_id"])
        if not customer:
            return {
                "content": [{"type": "text", "text": "Customer not found"}],
                "is_error": True,
            }
        request = {"requested_amount_usd": float(args["requested_amount"])}
        result = {
            "customer_credit_profile": {
                k: customer.get(k)
                for k in (
                    "credit_score",
                    "annual_income",
                    "income_verified",
                    "employment_status",
                    "current_credit_limit",
                    "credit_utilization",
                )
            },
            "derived": derive_credit_figures(request, customer),
        }
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error computing credit figures: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_policy",
    "Get the current policy rules for a specific category. Returns policy details including thresholds and requirements. If policy_name is provided, returns policies matching any words in the name.",
    {"category": str, "policy_name": str},
)
async def get_policy(args: dict[str, Any]) -> dict[str, Any]:
    """Get policy information."""
    try:
        # Get all policies for the category
        policies = context_graph_client.get_policies(category=args.get("category"))

        if args.get("policy_name"):
            # Extract meaningful words from the search query (skip common words)
            stop_words = {"the", "a", "an", "for", "and", "or", "of", "in", "to", "with"}
            search_words = [
                word.lower()
                for word in args["policy_name"].split()
                if word.lower() not in stop_words and len(word) > 2
            ]

            # Score each policy by how many search words match
            scored_policies = []
            for policy in policies:
                policy_name_lower = policy.get("name", "").lower()
                # Count how many search words appear in the policy name
                matches = sum(1 for word in search_words if word in policy_name_lower)
                if matches > 0:
                    scored_policies.append({"policy": policy, "relevance_score": matches})

            # Sort by relevance score (highest first)
            scored_policies.sort(key=lambda x: x["relevance_score"], reverse=True)

            if scored_policies:
                # Return all matching policies with relevance info
                results = {
                    "matching_policies": [
                        {**sp["policy"], "relevance_score": sp["relevance_score"]}
                        for sp in scored_policies
                    ],
                    "search_terms": search_words,
                    "total_matches": len(scored_policies),
                }
            else:
                # No matches found - return all policies in category as fallback
                results = {
                    "matching_policies": [],
                    "search_terms": search_words,
                    "total_matches": 0,
                    "all_policies_in_category": policies,
                    "note": f"No policies matched '{args['policy_name']}'. Showing all policies in category.",
                }
        else:
            results = policies

        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting policy: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "execute_cypher",
    "Execute a read-only Cypher query against the context graph for custom analysis. Only SELECT/MATCH queries are allowed.",
    {"cypher": str},
)
async def execute_cypher(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a read-only Cypher query."""
    try:
        results = context_graph_client.execute_cypher(cypher=args["cypher"])
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2, default=str)}]}
    except ValueError as e:
        return {
            "content": [{"type": "text", "text": f"Query not allowed: {str(e)}"}],
            "is_error": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error executing query: {str(e)}"}],
            "is_error": True,
        }


@tool(
    "get_schema",
    "Get the graph database schema including node labels, relationship types, property keys, indexes, and constraints. Also returns counts for each node label and relationship type.",
    {},
)
async def get_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Get the graph database schema."""
    try:
        schema = context_graph_client.get_schema()
        return {"content": [{"type": "text", "text": json.dumps(schema, indent=2, default=str)}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting schema: {str(e)}"}],
            "is_error": True,
        }


# ============================================
# MCP SERVER CREATION
# ============================================


def create_context_graph_server():
    """Create the MCP server with all context graph tools."""
    return create_sdk_mcp_server(
        name="context-graph",
        version="1.0.0",
        tools=[
            search_customer,
            get_customer_decisions,
            find_similar_decisions,
            find_precedents,
            get_causal_chain,
            record_decision,
            detect_fraud_patterns,
            find_decision_community,
            find_accounts_with_high_shared_transaction_volume,
            compute_credit_figures,
            get_policy,
            execute_cypher,
            get_schema,
        ],
    )


def get_agent_options(exclude_tools: tuple[str, ...] = ()) -> ClaudeAgentOptions:
    """
    Get the agent options with context graph server configured.

    ``exclude_tools`` removes tools by short name (e.g. "compute_credit_figures") so
    the eval harness can reproduce the original tool set exactly.
    """
    context_graph_server = create_context_graph_server()
    excluded = {f"mcp__graph__{name}" for name in exclude_tools}

    return ClaudeAgentOptions(
        system_prompt=CONTEXT_GRAPH_SYSTEM_PROMPT,
        model=config.anthropic.model or None,
        mcp_servers={"graph": context_graph_server},
        disallowed_tools=sorted(excluded),
        allowed_tools=[
            "mcp__graph__search_customer",
            "mcp__graph__get_customer_decisions",
            "mcp__graph__find_similar_decisions",
            "mcp__graph__find_precedents",
            "mcp__graph__get_causal_chain",
            "mcp__graph__record_decision",
            "mcp__graph__detect_fraud_patterns",
            "mcp__graph__find_decision_community",
            "mcp__graph__find_accounts_with_high_shared_transaction_volume",
            "mcp__graph__compute_credit_figures",
            "mcp__graph__get_policy",
            "mcp__graph__execute_cypher",
            "mcp__graph__get_schema",
        ],
    )


# Tools added after the original demo; excluded to reproduce the "original" eval arm.
POST_ORIGINAL_TOOLS = ("compute_credit_figures",)


# ============================================
# AGENT CONTEXT
# ============================================

AVAILABLE_TOOLS = [
    "search_customer",
    "get_customer_decisions",
    "find_similar_decisions",
    "find_precedents",
    "get_causal_chain",
    "record_decision",
    "detect_fraud_patterns",
    "find_decision_community",
    "find_accounts_with_high_shared_transaction_volume",
    "compute_credit_figures",
    "get_policy",
    "execute_cypher",
    "get_schema",
]


def get_agent_context() -> dict[str, Any]:
    """Get agent context information for transparency/debugging."""
    return {
        "system_prompt": CONTEXT_GRAPH_SYSTEM_PROMPT,
        "model": config.anthropic.model or "Agent SDK default (claude-sonnet-4-5)",
        "available_tools": AVAILABLE_TOOLS,
        "mcp_server": "context-graph",
    }


# ============================================
# AGENT SESSION MANAGEMENT
# ============================================


FINAL_RECOMMENDATION_RE = re.compile(
    r"FINAL_RECOMMENDATION:\s*\**\s*(approve|reject)\**"
    r"(?:\s*\(?\s*confidence[:\s]*([01](?:\.\d+)?)\s*\)?)?",
    re.IGNORECASE,
)


def extract_final_recommendation(text: str) -> tuple[str | None, float | None]:
    """Last FINAL_RECOMMENDATION line in ``text`` -> (approve|reject|None, confidence|None)."""
    match = None
    for match in FINAL_RECOMMENDATION_RE.finditer(text or ""):
        pass
    if not match:
        return None, None
    return match.group(1).lower(), (float(match.group(2)) if match.group(2) else None)


def parse_final_recommendation(text: str, fast_answer: dict[str, Any] | None) -> dict | None:
    """
    Pull Claude's FINAL_RECOMMENDATION line out of its reply and relate it to Jev's verdict.

    Returns None when there was no fast answer to relate it to.
    """
    if not fast_answer or not fast_answer.get("applicable"):
        return None
    verdict = fast_answer.get("verdict", {})
    jev_routed = verdict.get("routed_decision")
    escalated = jev_routed == "needs_llm_decision"

    match = None
    for match in FINAL_RECOMMENDATION_RE.finditer(text or ""):
        pass  # keep the last occurrence
    if not match:
        return {
            "decision": None,
            "confidence": None,
            "source": "claude" if escalated else "jev",
            "jev_decision": jev_routed,
            "escalated": escalated,
            "agrees_with_jev": None,
            "final": None if escalated else {"yes": "approve", "no": "reject"}.get(jev_routed),
        }

    decision = match.group(1).lower()
    confidence = float(match.group(2)) if match.group(2) else None
    jev_as_recommendation = {"yes": "approve", "no": "reject"}.get(jev_routed)
    return {
        "decision": decision,
        "confidence": confidence,
        "source": "claude" if escalated else "jev",
        "jev_decision": jev_routed,
        "escalated": escalated,
        "agrees_with_jev": None if escalated else decision == jev_as_recommendation,
        "final": decision,
    }


def format_fast_answer_context(fast_answer: dict[str, Any]) -> str:
    """Render the Jev fast answer and its evidence as context for Claude."""
    verdict = fast_answer.get("verdict", {})
    retrievals = ", ".join(r["name"] for r in fast_answer.get("retrievals", []))
    routed = verdict.get("routed_decision")
    if routed == "needs_llm_decision":
        task = (
            "Jev could not decide this case and escalated it to you. The user has been shown the "
            "escalation. Make the final call: should this request be approved or rejected? Weigh "
            "the evidence, name the precedents, policies and risk signals that drive your call, and "
            "say what would change your mind. Call tools only if something specific is missing from "
            "the evidence, and say what you needed. End with the FINAL_RECOMMENDATION line."
        )
    else:
        task = (
            f"The user has already been shown Jev's verdict ({routed}). Explain it: which precedents, "
            "policies and risk signals in the evidence support or complicate it, and what happens "
            "next. Confirm the verdict unless the evidence clearly contradicts it; if you disagree, "
            "say so and why. Call tools only if something specific is missing from the evidence, and "
            "say what you needed. End with the FINAL_RECOMMENDATION line."
        )
    return f"""<fast_answer>
A fast typed verdict was produced by TypeSafe's Jev model ({verdict.get("model")}) in {verdict.get("latency_ms")} ms
from evidence collected by a fixed set of Cypher retrievals ({retrievals}).

Request understood as: {json.dumps(fast_answer.get("request"), default=str)}
Resolved customer: {json.dumps(fast_answer.get("customer"), default=str)}

Jev verdict: {verdict.get("decision")}  (routed decision after confidence floor: {routed})
Probabilities: {json.dumps(verdict.get("probabilities"))}
Confidence: {verdict.get("confidence")}  Routing: {json.dumps(verdict.get("routing"))}
Supporting signals (probability each statement is true): {json.dumps(verdict.get("signals"))}

Pre-collected evidence (JSON):
{json.dumps(fast_answer.get("evidence"), indent=1, default=str)}
</fast_answer>

{task}"""


FINAL_LINE_INSTRUCTION = (
    "\n\nEnd your reply with exactly one line in this form, and nothing after it: "
    "`FINAL_RECOMMENDATION: approve (confidence 0.8)` or "
    "`FINAL_RECOMMENDATION: reject (confidence 0.7)`."
)


async def run_agent_only(
    message: str, exclude_tools: tuple[str, ...] = POST_ORIGINAL_TOOLS
) -> dict[str, Any]:
    """
    Tool-loop path: Claude picks tools and runs Cypher through them itself.
    By default the tool set is the original demo's (post-original tools excluded);
    pass ``exclude_tools=()`` to include the derived-figures tool (eval arm 2).
    Returns the response, the tool calls it made, and its parsed final recommendation.
    """
    import time

    started = time.perf_counter()
    async with ContextGraphAgent(exclude_tools=exclude_tools) as agent:
        result = await agent.query(message + FINAL_LINE_INSTRUCTION)
    decision, confidence = extract_final_recommendation(result["response"])
    tools = [c["name"].replace("mcp__graph__", "") for c in result["tool_calls"]]
    return {
        "path": "agent_only" if exclude_tools else "agent_with_derived_tool",
        "response": result["response"],
        "tool_calls": tools,
        "tool_call_count": len(tools),
        "final": decision,
        "confidence": confidence,
        "seconds": round(time.perf_counter() - started, 1),
    }


def format_evidence_only_context(case: dict[str, Any]) -> str:
    """The fixed evidence plan handed to Claude with no Jev verdict (eval arm 3)."""
    retrievals = ", ".join(r["name"] for r in case.get("retrievals", []))
    return f"""<evidence>
Evidence collected by a fixed set of Cypher retrievals ({retrievals}) run in code before you were
called. Treat it as already retrieved; do not re-run the same lookups. `request.derived` holds the
policy arithmetic (proposed total limit, its share of income, increase relative to the current
limit) computed by code.

Request: {json.dumps(case.get("request"), default=str)}
Resolved customer: {json.dumps(case.get("customer"), default=str)}

Evidence (JSON):
{json.dumps(case.get("evidence"), indent=1, default=str)}
</evidence>

Decide whether this request should be approved. Name the precedents, policies and risk signals that
drive your call. Call tools only if something specific is missing, and say what you needed.{FINAL_LINE_INSTRUCTION}"""


async def run_claude_on_evidence(message: str, case: dict[str, Any]) -> dict[str, Any]:
    """
    Eval arm 3: same fixed evidence as the fast path, Claude decides, no Jev.
    Isolates what code-owned evidence gathering is worth on its own.
    """
    import time

    started = time.perf_counter()
    text_parts: list[str] = []
    done: dict[str, Any] = {}
    full_message = format_evidence_only_context(case) + "\n\n" + message
    async with ContextGraphAgent() as agent:
        async for event in agent.query_stream(full_message):
            if event["type"] == "text":
                text_parts.append(event["content"])
            elif event["type"] == "done":
                done = event
    response = "\n".join(text_parts)
    decision, confidence = extract_final_recommendation(response)
    return {
        "path": "evidence_claude",
        "response": response,
        "tool_calls": done.get("additional_tools_used", []),
        "tool_call_count": len(done.get("additional_tools_used", [])),
        "final": decision,
        "confidence": confidence,
        "evidence_ms": case.get("evidence_ms"),
        "seconds": round(time.perf_counter() - started, 1),
    }


async def run_fast_answer_explanation(message: str, fast_answer: dict[str, Any]) -> dict[str, Any]:
    """
    The new path's second half: Claude explains (or, when escalated, decides) from
    the pre-collected evidence. Returns text, extra tools used and the final recommendation.
    """
    import time

    started = time.perf_counter()
    text_parts: list[str] = []
    done: dict[str, Any] = {}
    async with ContextGraphAgent() as agent:
        async for event in agent.query_stream(message, fast_answer=fast_answer):
            if event["type"] == "text":
                text_parts.append(event["content"])
            elif event["type"] == "done":
                done = event
    return {
        "path": "fast_answer",
        "response": "\n".join(text_parts),
        "tool_calls": done.get("additional_tools_used", []),
        "tool_call_count": len(done.get("additional_tools_used", [])),
        "final_recommendation": done.get("final_recommendation"),
        "final": (done.get("final_recommendation") or {}).get("final"),
        "confidence": (done.get("final_recommendation") or {}).get("confidence"),
        "seconds": round(time.perf_counter() - started, 1),
    }


class ContextGraphAgent:
    """Wrapper for managing Claude Agent SDK sessions."""

    def __init__(self, exclude_tools: tuple[str, ...] = ()):
        self.options = get_agent_options(exclude_tools)
        self.client: ClaudeSDKClient | None = None

    async def __aenter__(self):
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.disconnect()

    async def query(
        self, message: str, conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """Send a query to the agent and get the response."""
        if not self.client:
            raise RuntimeError("Agent not connected. Use 'async with' context manager.")

        # Build message with conversation context
        if conversation_history and len(conversation_history) > 0:
            # Format history as context in the message
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history[-6:]
                ]  # Last 6 messages
            )
            full_message = f"""Previous conversation:
{history_text}

Current message from USER: {message}

Please respond to the current message, taking the conversation history into account."""
        else:
            full_message = message

        # Send the message
        await self.client.query(full_message)

        response_text = ""
        tool_calls = []
        decisions_made = []

        async for msg in self.client.receive_response():
            # Process different message types
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text += block.text
                    elif hasattr(block, "name"):
                        # Tool use block
                        tool_calls.append(
                            {
                                "name": block.name,
                                "input": block.input if hasattr(block, "input") else {},
                            }
                        )
                        # Track decisions made
                        if block.name == "mcp__graph__record_decision":
                            # Will be populated when we get the result
                            pass

        return {
            "response": response_text,
            "tool_calls": tool_calls,
            "decisions_made": decisions_made,
        }

    async def query_stream(
        self,
        message: str,
        conversation_history: list[dict[str, str]] | None = None,
        fast_answer: dict[str, Any] | None = None,
    ):
        """
        Send a query to the agent and stream the response.

        ``fast_answer`` is the optional output of the Jev fast path: pre-collected
        evidence and a typed verdict. It is prepended to the message so Claude
        explains from the same context Jev decided on.
        """
        if not self.client:
            raise RuntimeError("Agent not connected. Use 'async with' context manager.")

        # Build message with conversation context
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in conversation_history[-6:]]
            )
            full_message = f"""Previous conversation:
{history_text}

Current message from USER: {message}

Please respond to the current message, taking the conversation history into account."""
        else:
            full_message = message

        if fast_answer and fast_answer.get("applicable"):
            full_message = format_fast_answer_context(fast_answer) + "\n\n" + full_message

        # Emit agent context first
        yield {"type": "agent_context", "context": get_agent_context()}

        # Send the message
        await self.client.query(full_message)

        tool_calls = []
        tool_id_to_name = {}  # Map tool_use_id to tool name
        decisions_made = []
        response_text = ""

        async for msg in self.client.receive_response():
            msg_type = type(msg).__name__

            # Handle UserMessage containing ToolResultBlock objects
            if msg_type == "UserMessage" and hasattr(msg, "content"):
                for block in msg.content:
                    block_type = type(block).__name__
                    # ToolResultBlock has tool_use_id and content attributes
                    if block_type == "ToolResultBlock":
                        tool_use_id = getattr(block, "tool_use_id", None)
                        block_content = getattr(block, "content", None)

                        print(f"[DEBUG] ToolResultBlock - tool_use_id: {tool_use_id}")

                        if tool_use_id:
                            parsed_output = None

                            # Parse the block content (list of content items)
                            if isinstance(block_content, list):
                                for item in block_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        try:
                                            parsed_output = json.loads(item.get("text", "{}"))
                                        except json.JSONDecodeError:
                                            parsed_output = item.get("text")
                                        break
                                    elif hasattr(item, "text"):
                                        try:
                                            parsed_output = json.loads(item.text)
                                        except json.JSONDecodeError:
                                            parsed_output = item.text
                                        break
                            elif isinstance(block_content, str):
                                try:
                                    parsed_output = json.loads(block_content)
                                except json.JSONDecodeError:
                                    parsed_output = block_content

                            # Look up the tool name from the tool_use_id
                            tool_name = tool_id_to_name.get(tool_use_id, "unknown")
                            print(
                                f"[DEBUG] Yielding tool_result: name={tool_name}, output_type={type(parsed_output)}"
                            )

                            yield {
                                "type": "tool_result",
                                "name": tool_name,
                                "output": parsed_output,
                            }
                continue

            # Handle AssistantMessage content blocks
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        # Stream text content
                        response_text += block.text + "\n"
                        yield {"type": "text", "content": block.text}
                    elif hasattr(block, "name"):
                        # Tool use block
                        tool_id = getattr(block, "id", None)
                        tool_call = {
                            "name": block.name,
                            "input": block.input if hasattr(block, "input") else {},
                        }
                        tool_calls.append(tool_call)

                        # Track tool_use_id to name mapping
                        if tool_id:
                            tool_id_to_name[tool_id] = block.name

                        yield {"type": "tool_use", **tool_call}

                        # Track decisions made
                        if block.name == "mcp__graph__record_decision":
                            pass

        # Final event with summary
        additional_tools = []
        for call in tool_calls:
            name = call["name"].replace("mcp__graph__", "")
            if name not in additional_tools:
                additional_tools.append(name)

        yield {
            "type": "done",
            "tool_calls": tool_calls,
            "decisions_made": decisions_made,
            "additional_tools_used": additional_tools,
            "evidence_retrievals": [r["name"] for r in (fast_answer or {}).get("retrievals", [])],
            "final_recommendation": parse_final_recommendation(response_text, fast_answer),
        }
